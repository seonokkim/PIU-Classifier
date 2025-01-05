# src/evaluate.py

from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import os

def evaluate_model(model, X_test, y_test, model_name, results_path):
    """
    Evaluate a given model and save the results.
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save results
    os.makedirs(results_path, exist_ok=True)
    report_df = pd.DataFrame(report).transpose()
    report_df["accuracy"] = accuracy
    report_file = os.path.join(results_path, f"{model_name}_results.csv")
    report_df.to_csv(report_file, index=True)

    print(f"\n{model_name} Accuracy: {accuracy:.3f}")
    print(f"Results saved to: {report_file}")

    return accuracy, report_file