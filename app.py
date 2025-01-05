import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


@st.cache_data
def get_clean_data():
    data_path = "./data/train.csv"
    data = pd.read_csv(data_path)

    # Drop columns containing 'Season'
    season_cols = [col for col in data.columns if "Season" in col]
    data = data.drop(season_cols, axis=1)

    # Impute missing values
    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(data[numeric_cols])
    train_imputed = pd.DataFrame(imputed_data, columns=numeric_cols)

    if "sii" in train_imputed.columns:
        train_imputed["sii"] = train_imputed["sii"].round().astype(int)

    for col in data.columns:
        if col not in numeric_cols:
            train_imputed[col] = data[col]

    if "id" in train_imputed.columns:
        train_imputed = train_imputed.drop("id", axis=1)

    return train_imputed


def add_sidebar():
    st.sidebar.header("Enter Patient Information")
    data = get_clean_data()

    input_features = [
        "Basic_Demos-Age",
        "Basic_Demos-Sex",
        "Physical-BMI",
        "Physical-Diastolic_BP",
        "Physical-HeartRate",
        "Physical-Systolic_BP",
    ]

    input_dict = {}
    for col in input_features:
        if col == "Basic_Demos-Age":
            input_dict[col] = st.sidebar.number_input(
                "Age",
                min_value=int(data[col].min()),
                max_value=int(data[col].max()),
                value=int(data[col].mean()),
            )
        elif col == "Basic_Demos-Sex":
            sex_value = st.sidebar.selectbox(
                "Sex (Male/Female)",
                options=["Male", "Female"],
                index=0 if data[col].mode()[0] == "Male" else 1,
            )
            input_dict[col] = 0 if sex_value == "Male" else 1
        else:
            input_dict[col] = st.sidebar.slider(
                col.replace("-", " ").replace("_", " "),
                min_value=float(data[col].min()),
                max_value=float(data[col].max()),
                value=float(data[col].mean()),
            )

    return input_dict


def prepare_full_input(input_data, data):
    feature_names = data.drop(columns=["sii"]).columns
    full_input_data = {}

    for feature in feature_names:
        full_input_data[feature] = input_data.get(feature, data[feature].mean())

    input_array = np.array([list(full_input_data.values())])
    return input_array


def add_predictions(input_data):
    # Load the model and scaler
    model = pickle.load(open("models/8_catboost_lightgbm_ensemble_model.pkl", "rb"))
    scaler = pickle.load(open("models/8_catboost_lightgbm_ensemble_scaler.pkl", "rb"))

    # Preprocess the input data
    data = get_clean_data()
    input_array = prepare_full_input(input_data, data)
    input_array_scaled = scaler.transform(input_array)

    # Display prediction results
    st.header("Prediction Results")
    st.markdown(
        "<hr style='border: 2px solid gray; border-radius: 5px;'>",
        unsafe_allow_html=True,
    )
    st.subheader("Severity Impairment Index (SII)")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_array_scaled)[0]

    severity = ""
    image_path = ""
    information = ""
    probability = int(probabilities[1] * 100)

    # Map probabilities to severity levels
    if probability <= 30:
        severity = "None"
        image_path = "./assets/None.png"
        information = (
            "SII indicates no significant dependency on internet use based on your data."
        )
    elif probability <= 49:
        severity = "Mild"
        image_path = "./assets/Mild.png"
        information = (
            "SII indicates mild dependency. Consider balancing your online and offline activities."
        )
    elif probability <= 79:
        severity = "Moderate"
        image_path = "./assets/Moderate.png"
        information = (
            "SII indicates moderate dependency. It is recommended to limit your internet usage."
        )
    else:
        severity = "Severe"
        image_path = "./assets/Severe.png"
        information = (
            "SII indicates severe dependency. Seek help from professionals if necessary."
        )

    colors = {
        "red": "#FF0000",
        "yellow": "#FFEC49",
        "green": "#33EFBA",
        "blue": "#34C9EF",
    }

    severity_color = {
        "Severe": colors["red"],
        "Moderate": colors["yellow"],
        "Mild": colors["green"],
        "None": colors["blue"],
    }

    st.image(image_path)
    st.markdown(
        f"<h3 style='text-align: center;'>Internet Dependency Level: <span style='color: {severity_color[severity]}'>{severity}</span> ({probability}%)</h3>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='padding: 10px; border-radius: 5px; background-color: #D9D9D9;'>{information}</div>",
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Internet Dependency Prediction Service",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        with open("assets/style.css") as f:
            st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    except FileNotFoundError:
        st.write("Style file not found. Proceeding with default styles.")

    input_data = add_sidebar()

    if st.sidebar.button("Run Prediction"):
        add_predictions(input_data)
    else:
        st.title("Internet Dependency Prediction")
        st.write("Provide patient information to predict internet dependency severity.")
        st.write(
            "Fill in the required fields on the sidebar and click 'Run Prediction' to see the results."
        )


if __name__ == "__main__":
    main()