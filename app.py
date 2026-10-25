import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Traffic Accident Severity Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "accident_severity_smote_model.pkl"


# =========================================================
# 3. PROFESSIONAL DASHBOARD DESIGN
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg,
            #F4F7FC 0%,
            #EAF1F8 100%
        );
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #17365D;
        text-align: center;
        margin-bottom: 8px;
    }

    .subtitle {
        font-size: 16px;
        color: #52657A;
        text-align: center;
        margin-bottom: 28px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #17365D;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    .info-card {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #DCE5EF;
        margin-bottom: 12px;
    }

    .result-card {
        background-color: white;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #DCE5EF;
        text-align: center;
        box-shadow: 0 3px 12px rgba(23, 54, 93, 0.07);
    }

    .result-number {
        font-size: 34px;
        font-weight: 800;
        color: #17365D;
    }

    .result-label {
        font-size: 15px;
        color: #52657A;
    }

    div.stButton > button {
        width: 100%;
        background-color: #17365D;
        color: white;
        font-weight: 700;
        border-radius: 9px;
        padding: 12px;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #087E8B;
        color: white;
    }

    [data-testid="stSidebar"] {
        background-color: #E7EEF7;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 4. LOAD THE SAVED MODEL
# =========================================================

@st.cache_resource
def load_model(model_path):
    saved_data = joblib.load(model_path)

    # The saved pickle contains a dictionary.
    if isinstance(saved_data, dict):

        if "model" not in saved_data:
            raise ValueError(
                "The saved file does not contain a 'model' entry."
            )

        if "feature_names" not in saved_data:
            raise ValueError(
                "The saved file does not contain 'feature_names'."
            )

        trained_model = saved_data["model"]
        saved_feature_names = saved_data["feature_names"]

    else:
        trained_model = saved_data

        if hasattr(trained_model, "feature_names_in_"):
            saved_feature_names = list(
                trained_model.feature_names_in_
            )
        else:
            raise ValueError(
                "The original model feature names could not be found."
            )

    return trained_model, list(saved_feature_names)


if not MODEL_PATH.exists():
    st.error(
        "The trained model file was not found.\n\n"
        f"Expected location: {MODEL_PATH}"
    )
    st.stop()


try:
    model, feature_names = load_model(str(MODEL_PATH))

except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# =========================================================
# 5. DASHBOARD HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🚗 Traffic Accident Severity Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning-Based Accident Severity Analysis'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Enter the available weather, location, time, and road-condition "
    "details to obtain a model-generated accident severity prediction. "
    "This application is an academic prototype, not an emergency "
    "response or road-safety decision system."
)


# =========================================================
# 6. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🚘 Project Information")

    st.write(
        "This application uses a trained Random Forest classifier "
        "to predict one of four accident severity classes."
    )

    st.markdown("---")

    st.markdown("### Severity Levels")

    st.write("**Level 1:** Lowest severity class")
    st.write("**Level 2:** Second severity class")
    st.write("**Level 3:** Third severity class")
    st.write("**Level 4:** Highest severity class")

    st.caption(
        "These labels describe the dataset's severity classes. "
        "They do not guarantee the real-world seriousness of an accident."
    )

    st.markdown("---")

    st.markdown("### Model")

    st.write("SMOTE Random Forest")

    st.write("Input features:", len(feature_names))


# =========================================================
# 7. INPUT FORM
# =========================================================

st.markdown(
    '<div class="section-title">📍 Accident Details</div>',
    unsafe_allow_html=True
)

with st.form("accident_prediction_form"):

    # -----------------------------------------------------
    # LOCATION
    # -----------------------------------------------------

    st.subheader("📍 Location")

    col1, col2 = st.columns(2)

    with col1:
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=16.5062,
            step=0.0001,
            format="%.4f"
        )

    with col2:
        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=80.6480,
            step=0.0001,
            format="%.4f"
        )

    st.markdown("---")

    # -----------------------------------------------------
    # WEATHER
    # -----------------------------------------------------

    st.subheader("🌦️ Weather Conditions")

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input(
            "Temperature (°F)",
            value=30.0,
            step=1.0
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=65.0,
            step=1.0
        )

        pressure = st.number_input(
            "Pressure (in)",
            min_value=0.0,
            value=29.9,
            step=0.1
        )

        visibility = st.number_input(
            "Visibility (miles)",
            min_value=0.0,
            value=10.0,
            step=0.5
        )

    with col2:
        wind_speed = st.number_input(
            "Wind Speed (mph)",
            min_value=0.0,
            value=5.0,
            step=1.0
        )

        precipitation = st.number_input(
            "Precipitation (in)",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

        weather_features = [
            name for name in feature_names
            if name.startswith("Weather_Condition_")
        ]

        weather_options = sorted(
            [
                name[len("Weather_Condition_"):]
                for name in weather_features
            ]
        )

        if weather_options:
            default_weather = (
                "Clear" if "Clear" in weather_options
                else weather_options[0]
            )

            weather_index = weather_options.index(default_weather)

            weather = st.selectbox(
                "Weather Condition",
                options=weather_options,
                index=weather_index
            )

        else:
            weather = st.selectbox(
                "Weather Condition",
                options=["Clear", "Cloudy", "Rain", "Fog"]
            )

    st.markdown("---")

    # -----------------------------------------------------
    # TIME
    # -----------------------------------------------------

    st.subheader("🕒 Time Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        hour = st.slider(
            "Hour of Day",
            min_value=0,
            max_value=23,
            value=14
        )

    with col2:
        day_of_week = st.selectbox(
            "Day of Week",
            options=list(range(7)),
            index=2,
            format_func=lambda x: [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ][x]
        )

    with col3:
        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            index=9,
            format_func=lambda x: [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December"
            ][x - 1]
        )

    # -----------------------------------------------------
    # SUNRISE / SUNSET
    # -----------------------------------------------------

    sunrise_sunset_features = [
        name for name in feature_names
        if name.startswith("Sunrise_Sunset_")
    ]

    sunrise_options = sorted(
        [
            name[len("Sunrise_Sunset_"):]
            for name in sunrise_sunset_features
        ]
    )

    civil_twilight_features = [
        name for name in feature_names
        if name.startswith("Civil_Twilight_")
    ]

    civil_options = sorted(
        [
            name[len("Civil_Twilight_"):]
            for name in civil_twilight_features
        ]
    )

    col1, col2 = st.columns(2)

    with col1:
        if sunrise_options:
            sunrise_sunset = st.selectbox(
                "Sunrise / Sunset",
                options=sunrise_options,
                index=(
                    sunrise_options.index("Day")
                    if "Day" in sunrise_options
                    else 0
                )
            )
        else:
            sunrise_sunset = "Day"

    with col2:
        if civil_options:
            civil_twilight = st.selectbox(
                "Civil Twilight",
                options=civil_options,
                index=(
                    civil_options.index("Day")
                    if "Day" in civil_options
                    else 0
                )
            )
        else:
            civil_twilight = "Day"

    st.markdown("---")

    # -----------------------------------------------------
    # ROAD CONDITIONS
    # -----------------------------------------------------

    st.subheader("🛣️ Road Conditions")

    road_feature_names = [
        "Amenity",
        "Bump",
        "Crossing",
        "Give_Way",
        "Junction",
        "No_Exit",
        "Railway",
        "Roundabout",
        "Station",
        "Stop",
        "Traffic_Calming",
        "Traffic_Signal",
        "Turning_Loop"
    ]

    road_labels = {
        "Amenity": "Amenity nearby",
        "Bump": "Road bump",
        "Crossing": "Pedestrian crossing",
        "Give_Way": "Give-way sign",
        "Junction": "Junction",
        "No_Exit": "No-exit road",
        "Railway": "Railway crossing",
        "Roundabout": "Roundabout",
        "Station": "Station nearby",
        "Stop": "Stop sign",
        "Traffic_Calming": "Traffic calming",
        "Traffic_Signal": "Traffic signal",
        "Turning_Loop": "Turning loop"
    }

    road_values = {}

    road_columns = st.columns(3)

    for index, feature in enumerate(road_feature_names):

        with road_columns[index % 3]:

            if feature in feature_names:
                road_values[feature] = st.checkbox(
                    road_labels[feature],
                    value=False,
                    key=f"road_{feature}"
                )

            else:
                road_values[feature] = False

    st.markdown("---")

    submitted = st.form_submit_button(
        "🔮 Predict Accident Severity"
    )


# =========================================================
# 8. PREPARE INPUT DATA
# =========================================================

if submitted:

    try:
        # Create a row with exactly the features used during training.
        input_data = pd.DataFrame(
            [[0.0] * len(feature_names)],
            columns=feature_names,
            dtype=float
        )

        def set_numeric_feature(name, value):
            if name in input_data.columns:
                input_data.at[0, name] = value

        def set_category_feature(prefix, category):
            """
            Set the corresponding one-hot encoded category.
            All other categories in the same group remain zero.
            """

            matching_columns = [
                name for name in feature_names
                if name.startswith(prefix + "_")
            ]

            for column in matching_columns:
                input_data.at[0, column] = 0

            selected_column = f"{prefix}_{category}"

            if selected_column in input_data.columns:
                input_data.at[0, selected_column] = 1

        # Numeric features
        set_numeric_feature("Start_Lat", latitude)
        set_numeric_feature("Start_Lng", longitude)
        set_numeric_feature("Temperature(F)", temperature)
        set_numeric_feature("Humidity(%)", humidity)
        set_numeric_feature("Pressure(in)", pressure)
        set_numeric_feature("Visibility(mi)", visibility)
        set_numeric_feature("Wind_Speed(mph)", wind_speed)
        set_numeric_feature("Precipitation(in)", precipitation)
        set_numeric_feature("Hour", hour)
        set_numeric_feature("DayOfWeek", day_of_week)
        set_numeric_feature("Month", month)

        # Categorical features
        set_category_feature(
            "Weather_Condition",
            weather
        )

        set_category_feature(
            "Sunrise_Sunset",
            sunrise_sunset
        )

        set_category_feature(
            "Civil_Twilight",
            civil_twilight
        )

        # Road condition features
        for feature, value in road_values.items():
            set_numeric_feature(
                feature,
                int(value)
            )

        # Ensure feature order matches the trained model.
        input_data = input_data[feature_names]

        # Convert feature values to numeric types.
        input_data = input_data.apply(
            pd.to_numeric,
            errors="coerce"
        ).fillna(0)

        # =================================================
        # 9. MAKE PREDICTION
        # =================================================

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        classes = list(model.classes_)

        probability_df = pd.DataFrame({
            "Severity": [
                f"Level {int(class_label)}"
                for class_label in classes
            ],
            "Probability": probabilities
        })

        probability_df["Probability (%)"] = (
            probability_df["Probability"] * 100
        ).round(2)

        probability_df = probability_df[
            ["Severity", "Probability (%)"]
        ]

        # =================================================
        # 10. DISPLAY PREDICTION
        # =================================================

        st.markdown("---")

        st.markdown(
            '<div class="section-title">'
            '📊 Prediction Results'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Predicted Severity
                    </div>
                    <div class="result-number">
                        Level {int(prediction)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            predicted_probability = probabilities[
                classes.index(prediction)
            ] * 100

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Model Probability for Predicted Class
                    </div>
                    <div class="result-number">
                        {predicted_probability:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        severity_descriptions = {
            1: "The model predicts severity class 1.",
            2: "The model predicts severity class 2.",
            3: "The model predicts severity class 3.",
            4: "The model predicts severity class 4."
        }

        st.info(
            severity_descriptions.get(
                int(prediction),
                "The model returned a severity class."
            )
        )

        st.caption(
            "The displayed probability is the model's estimated "
            "probability for the predicted class. It is not a guarantee "
            "of real-world accident severity or a measure of model accuracy."
        )

        # =================================================
        # 11. PROBABILITY CHART
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '📈 Severity Probability Distribution'
            '</div>',
            unsafe_allow_html=True
        )

        fig = px.bar(
            probability_df,
            x="Severity",
            y="Probability (%)",
            text="Probability (%)",
            title="Predicted Probability for Each Severity Class",
            color="Severity",
            color_discrete_sequence=[
                "#43AA8B",
                "#277DA1",
                "#F9C74F",
                "#F94144"
            ]
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            template="plotly_white",
            yaxis_title="Probability (%)",
            xaxis_title="Severity Class",
            yaxis_range=[
                0,
                max(
                    100,
                    float(probability_df["Probability (%)"].max()) + 10
                )
            ],
            showlegend=False,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =================================================
        # 12. PROBABILITY TABLE
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '📋 Prediction Probability Table'
            '</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )

        # =================================================
        # 13. INPUT SUMMARY
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '📝 Input Summary'
            '</div>',
            unsafe_allow_html=True
        )

        summary_data = {
            "Input Feature": [
                "Latitude",
                "Longitude",
                "Temperature (°F)",
                "Humidity (%)",
                "Pressure (in)",
                "Visibility (miles)",
                "Wind Speed (mph)",
                "Precipitation (in)",
                "Weather Condition",
                "Hour",
                "Day of Week",
                "Month",
                "Sunrise / Sunset",
                "Civil Twilight"
            ],
            "Value": [
                latitude,
                longitude,
                temperature,
                humidity,
                pressure,
                visibility,
                wind_speed,
                precipitation,
                weather,
                hour,
                day_of_week,
                month,
                sunrise_sunset,
                civil_twilight
            ]
        }

        # Fix for mixed numeric and text values in Streamlit tables.
        summary_df = pd.DataFrame(summary_data)

        summary_df["Value"] = (
            summary_df["Value"].astype(str)
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

        # =================================================
        # 14. DOWNLOAD PREDICTION
        # =================================================

        download_data = probability_df.copy()

        download_data["Predicted Severity"] = (
            f"Level {int(prediction)}"
        )

        csv_data = download_data.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Prediction Results",
            data=csv_data,
            file_name="accident_severity_prediction.csv",
            mime="text/csv"
        )

    except Exception as error:

        st.error(
            "An error occurred while generating the prediction."
        )

        st.exception(error)


# =========================================================
# 15. FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#52657A; padding:10px;">
        <b>Traffic Accident Severity Prediction</b><br>
        MSc Computational Data Science Project<br>
        Developed using Python, Scikit-learn, Pandas and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
