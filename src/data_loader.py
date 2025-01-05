# src/data_loader.py

import pandas as pd
from sklearn.impute import KNNImputer

def get_clean_data(data_path):
    """
    Load and preprocess the data.
    - Drop columns containing 'Season'.
    - Apply KNN Imputation for missing values.
    - Drop the 'id' column if it exists.
    """
    data = pd.read_csv(data_path)

    # Drop columns with 'Season' in their name
    season_cols = [col for col in data.columns if "Season" in col]
    data = data.drop(season_cols, axis=1)

    # Identify numeric columns
    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns

    # Apply KNN Imputer
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(data[numeric_cols])
    data_imputed = pd.DataFrame(imputed_data, columns=numeric_cols)

    # Restore non-numeric columns
    for col in data.columns:
        if col not in numeric_cols:
            data_imputed[col] = data[col]

    # Drop the 'id' column
    if "id" in data_imputed.columns:
        data_imputed = data_imputed.drop("id", axis=1)

    return data_imputed