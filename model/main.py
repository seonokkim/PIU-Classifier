import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import pickle


def get_clean_data(data_path):
    """
    데이터셋을 로드하고 전처리 및 클리닝을 수행하는 함수
    """
    # 데이터 로드
    data = pd.read_csv(data_path)
    
    # 'Season'을 포함하는 열 제거
    season_cols = [col for col in data.columns if 'Season' in col]
    data = data.drop(season_cols, axis=1)
    
    # 숫자형 데이터 열 식별
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    
    # KNN Imputer를 사용하여 숫자형 데이터의 결측값 처리
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(data[numeric_cols])
    
    # 처리된 데이터를 데이터프레임으로 변환
    train_imputed = pd.DataFrame(imputed_data, columns=numeric_cols)
    
    # 'sii' 열이 존재하면 정수형으로 변환
    if 'sii' in train_imputed.columns:
        train_imputed['sii'] = train_imputed['sii'].round().astype(int)
    
    # 숫자형이 아닌 열을 원래 데이터프레임에서 복원
    for col in data.columns:
        if col not in numeric_cols:
            train_imputed[col] = data[col]
    
    # 'id' 열이 존재하면 제거
    if 'id' in train_imputed.columns:
        train_imputed = train_imputed.drop('id', axis=1)
    
    return train_imputed


def create_model(data, epochs=100):
    """
    Logistic Regression 모델을 학습하고, 학습 손실을 추적하는 함수
    """
    # 특징(X)와 타겟(y) 분리
    X = data.drop(['sii'], axis=1)
    y = data['sii']
    
    # 데이터 스케일링
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 학습 데이터와 테스트 데이터로 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Logistic Regression 모델 초기화 (saga solver 사용)
    model = LogisticRegression(solver='saga', max_iter=1)
    
    # 학습 손실을 추적할 리스트
    train_losses = []
    
    # 에포크 반복
    for epoch in range(epochs):
        model.fit(X_train, y_train)  # 학습 수행
        y_train_pred_proba = model.predict_proba(X_train)  # 학습 데이터에 대한 확률 예측
        loss = log_loss(y_train, y_train_pred_proba)  # 손실 계산
        train_losses.append(loss)  # 손실 저장
    
    # 학습 손실 시각화
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.title("Logistic Regression Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.grid()
    plt.legend()
    plt.show()
    
    # 테스트 데이터 평가
    y_pred = model.predict(X_test)
    print('모델 정확도: ', accuracy_score(y_test, y_pred))
    print("분류 리포트: \n", classification_report(y_test, y_pred))
    
    return model, scaler


def save_model_and_scaler(model, scaler, model_path='model.pkl', scaler_path='scaler.pkl'):
    """
    학습된 모델과 스케일러를 pickle 파일로 저장하는 함수
    """
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    with open(scaler_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    print(f"모델이 {model_path}에 저장되었습니다.")
    print(f"스케일러가 {scaler_path}에 저장되었습니다.")


def main():
    """
    데이터를 로드하고, 모델을 학습하고, 결과를 저장하는 메인 함수
    """
    # 데이터셋 경로 정의
    data_path = "train.csv"
    
    # 데이터 전처리 및 클리닝
    data = get_clean_data(data_path)
    
    # 모델 학습 및 스케일러 반환
    model, scaler = create_model(data, epochs=100)
    
    # 모델과 스케일러 저장
    save_model_and_scaler(model, scaler)


if __name__ == '__main__':
    main()
