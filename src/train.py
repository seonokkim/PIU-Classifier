# src/train.py

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

def train_models(data, model_path):
    """
    Train CatBoost, LightGBM, and an ensemble model.
    Save the trained models and scaler to the specified path.
    """
    # Separate features and target
    X = data.drop("sii", axis=1)
    y = data["sii"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train CatBoost model
    catboost_model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, random_seed=42, verbose=0)
    catboost_model.fit(X_train_scaled, y_train)

    # Train LightGBM model
    lightgbm_model = LGBMClassifier(n_estimators=500, learning_rate=0.1, max_depth=6, random_state=42)
    lightgbm_model.fit(X_train_scaled, y_train)

    # Create ensemble model
    ensemble_model = VotingClassifier(estimators=[
        ("catboost", catboost_model),
        ("lightgbm", lightgbm_model)
    ], voting="soft")
    ensemble_model.fit(X_train_scaled, y_train)

    # Save models and scaler
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "catboost_model.pkl"), "wb") as f:
        pickle.dump(catboost_model, f)
    with open(os.path.join(model_path, "lightgbm_model.pkl"), "wb") as f:
        pickle.dump(lightgbm_model, f)
    with open(os.path.join(model_path, "ensemble_model.pkl"), "wb") as f:
        pickle.dump(ensemble_model, f)
    with open(os.path.join(model_path, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    return catboost_model, lightgbm_model, ensemble_model, scaler