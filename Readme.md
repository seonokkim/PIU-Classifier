# Logistic Regression x Streamlit

This repository provides a `Streamlit` web application to predict the 'sii' metric based on a dataset related to various health and demographic measurements. The application utilizes `Logistic Regression` as the predictive model and integrates `KNN Imputation` for handling missing data.

## Demo

You can check a live demo of the application and further details [here](https://youtu.be/1xjrEED2PS4).

## Video Guide

The project was inspired by this [YouTube tutorial](https://www.youtube.com/watch?v=NfwfiyMi1lk&embeds_referring_euri=https%3A%2F%2Fwww.notion.so%2F&source_ve_path=MjM4NTE), which provides a step-by-step guide on implementing Logistic Regression with Streamlit.

## Features

- **Data Preprocessing**: 
  - Uses `KNNImputer` to handle missing values in numeric columns.
  - Drops irrelevant columns (e.g., containing 'Season') and identifier columns (`id`) if present.
  
- **Model**: 
  - Implements a `Logistic Regression` classifier to predict 'sii'.
  - Data scaling is performed using `StandardScaler`.

- **Streamlit Application**:
  - Interactive sidebar with sliders for all 69 input features, excluding the target variable ('sii').
  - Generates predictions dynamically based on user inputs.

## Installation

Ensure you have Python installed, then clone this repository and install the dependencies:

```bash
git clone https://github.com/your-repo-name.git
cd your-repo-name
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

The model achieved an accuracy of **0.93**. Below is the detailed classification report:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.95      | 0.96   | 0.96     | 412     |
| 1     | 0.91      | 0.92   | 0.91     | 285     |
| 2     | 0.90      | 0.87   | 0.89     | 87      |
| 3     | 0.67      | 0.50   | 0.57     | 8       |

**Overall Metrics**:

| Metric           | Value |
|------------------|-------|
| Accuracy         | 0.93  |
| Macro Avg (F1)   | 0.83  |
| Weighted Avg (F1)| 0.93  |
