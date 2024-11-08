import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


@st.cache_data
def get_clean_data():
    data_path = "./data/train.csv"
    data = pd.read_csv(data_path)

    season_cols = [col for col in data.columns if "Season" in col]
    data = data.drop(season_cols, axis=1)

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
    st.sidebar.header("학생 정보를 입력해주세요 :angel:")
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
                "나이",
                min_value=int(data[col].min()),
                max_value=int(data[col].max()),
                value=int(data[col].mean()),
            )
        elif col == "Basic_Demos-Sex":
            sex_value = st.sidebar.selectbox(
                "성별 (Male/Female)",
                options=["Male", "Female"],
                index=0 if data[col].mode()[0] == "Male" else 1,
            )
            input_dict[col] = 0 if sex_value == "Male" else 1
        elif col == "Physical-Height":
            height_in_cm = (
                float(data[col].mean()) * 2.54
            )  # Convert mean from inches to cm
            min_height_cm = (
                float(data[col].min()) * 2.54
            )  # Convert min from inches to cm
            max_height_cm = (
                float(data[col].max()) * 2.54
            )  # Convert max from inches to cm

            height_input_cm = st.sidebar.number_input(
                "키 (cm)",
                min_value=min_height_cm,
                max_value=max_height_cm,
                value=height_in_cm,
            )
            input_dict[col] = height_input_cm / 2.54
        elif col == "Physical-Weight":
            weight_in_kg = (
                float(data[col].mean()) * 0.45359237
            )  # Convert mean from lbs to kg
            min_weight_kg = (
                float(data[col].min()) * 0.45359237
            )  # Convert min from lbs to kg
            max_weight_kg = (
                float(data[col].max()) * 0.45359237
            )  # Convert max from lbs to kg

            weight_input_kg = st.sidebar.number_input(
                "몸무게 (kg)",
                min_value=min_weight_kg,
                max_value=max_weight_kg,
                value=weight_in_kg,
            )
            input_dict[col] = weight_input_kg / 0.45359237
        elif col == "Physical-BMI":
            input_dict[col] = st.sidebar.slider(
                "BMI 지수",
                min_value=float(data[col].min()),
                max_value=float(data[col].max()),
                value=float(data[col].mean()),
            )
        elif col == "Physical-Diastolic_BP":
            input_dict[col] = st.sidebar.slider(
                "혈압 (Diastolic)",
                min_value=float(data[col].min()),
                max_value=float(data[col].max()),
                value=float(data[col].mean()),
            )
        elif col == "Physical-HeartRate":
            input_dict[col] = st.sidebar.slider(
                "심박수",
                min_value=float(data[col].min()),
                max_value=float(data[col].max()),
                value=float(data[col].mean()),
            )
        elif col == "Physical-Systolic_BP":
            input_dict[col] = st.sidebar.slider(
                "혈압 (Systolic)",
                min_value=float(data[col].min()),
                max_value=float(data[col].max()),
                value=float(data[col].mean()),
            )
        else:
            input_dict[col] = st.sidebar.slider(
                col,
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
    print("add_predictions", input_data)
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    data = get_clean_data()
    input_array = prepare_full_input(input_data, data)
    input_array_scaled = scaler.transform(input_array)

    st.header("검사 결과")
    st.markdown(
        "<hr style='border: 2px solid gray; border-radius: 5px;'>",
        unsafe_allow_html=True,
    )
    st.subheader("심각도 장애 지수 (Severity Impairment Index, SII)")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_array_scaled)[0]

    # Severity Impairment Index: 0-30=None; 31-49=Mild; 50-79=Moderate; 80-100=Severe
    severity = ""
    image_path = ""
    information = ""
    probability = int(probabilities[1] * 100)

    if probability <= 30:
        severity = "정상"
        image_path = "./assets/None.png"
        information = "Severity Impairment Index(SII)는 사용자의 신체 데이터를 사용하여 인터넷 의존도를 체크하는 지표입니다. 검사 결과 정상에 해당하면, 사용자는 생체 및 신체 데이터 및 활동량 데이터를 분석했을 때 인터넷 중독 정도에 문제가 없음을 뜻합니다."

    elif probability <= 49:
        severity = "경도"
        image_path = "./assets/Mild.png"
        information = "Severity Impairment Index(SII)는 사용자의 신체 데이터를 사용하여 인터넷 의존도를 체크하는 지표입니다. 검사 결과 경도에 해당하면, 사용자는 생체 및 신체 데이터 및 활동량 데이터를 분석했을 때 인터넷 중독 정도가 평균 이하-평균 수준임을 뜻합니다. 인터넷 과의존 증상을 보이지는 않지만, 신체 활동량을 늘리고 인터넷 사용량을 줄이려는 노력이 어느 정도 필요합니다."
    elif probability <= 79:
        severity = "중등도"
        image_path = "./assets/Moderate.png"
        information = "Severity Impairment Index(SII)는 사용자의 신체 데이터를 사용하여 인터넷 의존도를 체크하는 지표입니다. 검사 결과 중등도에 해당하면, 사용자는 생체 및 신체 데이터 및 활동량 데이터를 분석했을 때 인터넷 중독 정도가 평균-평균 이상 수준임을 뜻합니다. 인터넷 과의존 증상이 있을 수 있고, 신체 활동량을 늘리고 인터넷 사용량을 줄이려는 노력이 반드시 필요합니다. 필요하다면 전문가의 도움을 받을 수 있습니다."
    else:
        severity = "중증"
        image_path = "./assets/Severe.png"
        information = "Severity Impairment Index(SII)는 사용자의 신체 데이터를 사용하여 인터넷 의존도를 체크하는 지표입니다. 검사 결과 중증에 해당하면, 사용자는 생체 및 신체 데이터 및 활동량 데이터를 분석했을 때 인터넷 중독 정도가 심각한 수준임을 뜻합니다. 인터넷 중독 증상이 분명히 있으며, 신체 활동량을 늘리고 인터넷 사용량을 줄이지 않으면 건강에 심각한 영향을 미칠 수 있습니다. 가족 구성원과 전문가의 도움을 강력히 권고합니다."

    colors = {
        "red": "#FF0000",
        "yellow": "#FFEC49",
        "green": "#33EFBA",
        "blue": "#34C9EF",
    }

    severity_color = {
        "중증": colors["red"],
        "중등도": colors["yellow"],
        "경도": colors["green"],
        "정상": colors["blue"],
    }

    st.image(image_path)
    st.write("")
    st.write("")
    st.write("")

    st.markdown(
        f"<h3 style='text-align: center'>인터넷 의존 단계가 <span style='color: {severity_color[severity]}'>{severity}</span><span style='color: {severity_color[severity]}'>({probability})</span> 입니다.</h3>",
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("")
    st.markdown(
        f"<div style='padding: 10px; border-radius: 5px; background-color: #D9D9D9;'><span style='color: white;'>{information}</span></div>",
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("")
    st.write("")
    st.markdown(
        "<p style='text-align: center; color: gray;'>본 결과는 건강정보에 대한 참고자료일 뿐이며, 정확한 판단을 위해서는 전문가의 진료가 반드시 필요합니다</p>",
        unsafe_allow_html=True,
    )

    if probability <= 30:
        st.balloons()


def main():
    st.set_page_config(
        page_title="인터넷 중증도 예측 서비스",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        with open("assets/style.css") as f:
            st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    except FileNotFoundError:
        st.write("스타일 파일을 찾을 수 없어 기본 스타일로 진행합니다.")

    input_data = add_sidebar()

    if st.sidebar.button("예측 실행"):
        add_predictions(input_data)
    else:
        st.title("인터넷 중증도 예측")
        st.write("")
        st.write("아이에 관한 정보를 입력하면 중증도 예측을 수행하는 앱입니다.")
        st.write(
            "왼쪽 사이드바에서 정보를 입력하고 '예측 실행' 버튼을 눌러 결과를 확인하세요."
        )


if __name__ == "__main__":
    main()
