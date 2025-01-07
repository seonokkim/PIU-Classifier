
# PIU-Classifier: Predicting Problematic Internet Use Severity
![AdaptCoder Diagram](https://github.com/seonokkim/AdaptCoder/blob/main/figure/AdaptCoder.jpg)

This project focuses on developing and deploying machine learning models to predict the ‘sii’ metric, a severity index for problematic internet use. Leveraging advanced ML techniques, it applies CatBoost, LightGBM, and Voting Classifier models to analyze health and demographic data. The pipeline incorporates KNN Imputation to handle missing values and uses scalable, robust preprocessing to ensure accurate predictions.

The trained models are integrated into a Streamlit web application, allowing users to input health and demographic data interactively and receive real-time predictions of the ‘sii’ metric. This application provides an accessible and intuitive interface for practical use.

## Demo

Check a live demo of the application and further details [here](https://youtu.be/1xjrEED2PS4).

## Video Guide

The project was inspired by this [YouTube tutorial](https://www.youtube.com/watch?v=NfwfiyMi1lk&embeds_referring_euri=https%3A%2F%2Fwww.notion.so%2F&source_ve_path=MjM4NTE), which provides a step-by-step guide on implementing Logistic Regression with Streamlit.

## Features

- **Data Preprocessing**: 
  - Uses `KNNImputer` to handle missing values in numeric columns.
  - Drops irrelevant columns (e.g., containing 'Season') and identifier columns (`id`) if present.
  
- **Models**: 
  - Implements a range of machine learning models:
    - Logistic Regression, XGBoost, CatBoost, LightGBM, Gradient Boosting, and ensemble methods.
  - Data scaling is performed using `StandardScaler` for all models.

- **Streamlit Application**:
  - Interactive sidebar with sliders for input features, excluding the target variable ('sii').
  - Generates predictions dynamically based on user inputs.

## Installation

Ensure you have Python installed, then clone this repository and install the dependencies:

```bash
git clone https://github.com/seonokkim/PIU-Classifier.git
cd PIU-Classifier
pip install -r requirements.txt
```

## Usage

### Data Preparation

Place the training dataset in the designated directory (`/data/train.csv`).  
The dataset can be downloaded from the [Kaggle competition: Child Mind Institute Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data).

### Training

Run `train.py` (or an equivalent script) to process the data, train the model, and save the model and scaler.

### Running the Application

Start the Streamlit app by executing:

```bash
streamlit run app.py
```

## Test Results

The performance of various models is summarized in the table below:

| Model                          | Type      | Precision | Recall | F1-Score | Accuracy |
|--------------------------------|-----------|-----------|--------|----------|----------|
| Logistic Regression            | Single    | 0.983     | 0.982  | 0.982    | 0.977    |
| XGBoost                        | Single    | 0.983     | 0.982  | 0.982    | 0.977    |
| AdaBoost                       | Single    | 0.973     | 0.990  | 0.982    | 0.977    |
| Gradient Boosting              | Single    | 0.976     | 0.993  | 0.984    | 0.979    |
| AdaBoost + Gradient Boosting   | Ensemble  | 0.976     | 0.993  | 0.984    | 0.979    |
| CatBoost                       | Single    | 0.983     | 0.984  | 0.983    | 0.977    |
| LightGBM                       | Single    | 0.986     | 0.987  | 0.986    | 0.980    |
| CatBoost + LightGBM            | Ensemble  | 0.976     | 0.993  | 0.984    | 0.981    |

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
