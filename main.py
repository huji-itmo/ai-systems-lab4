import numpy as np
import pandas as pd

from analyze_dataset import analyze_dataset
from regression import multiple_linear_regression_scalar


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical 'Extracurricular Activities' to binary."""
    df = df.copy()
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map(
        {"Yes": 1, "No": 0}
    )
    return df


def main(csv_path: str = "data.csv"):
    # Load data
    df = pd.read_csv(csv_path)

    # Preprocess
    df = preprocess_features(df)

    # Define features and target
    feature_columns = [
        "Hours Studied",
        "Previous Scores",
        "Extracurricular Activities",
        "Sleep Hours",
        "Sample Question Papers Practiced",
    ]
    target_column = "Performance Index"

    X = df[feature_columns].values
    y = np.array(df[target_column].values)

    # Fit model
    coeffs = multiple_linear_regression_scalar(y, X)
    intercept = coeffs[0]
    betas = coeffs[1:]

    print("✅ Model trained successfully!")
    print(f"Intercept (a): {intercept:.4f}")
    for name, beta in zip(feature_columns, betas):
        print(f"Coefficient for '{name}': {beta:.4f}")

    # --- Compute predictions on training data ---
    X_with_const = np.column_stack([np.ones(X.shape[0]), X])
    y_pred = X_with_const @ coeffs

    # --- Compute R² (coefficient of determination) ---
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # --- Compute Pearson correlation coefficient R (without scipy) ---
    # Using np.corrcoef — returns a 2x2 matrix; [0,1] is the correlation
    r = np.corrcoef(y, y_pred)[0, 1]

    print(f"\n📊 Model Performance:")
    print(f"R² (Coefficient of Determination): {r_squared:.4f}")
    print(f"R (Correlation Coefficient): {r:.4f}")

    # Example prediction
    print("\n--- Predict Performance Index ---")
    try:
        hours_studied = float(input("Hours Studied: "))
        prev_scores = float(input("Previous Scores: "))
        extra = input("Extracurricular Activities (Yes/No): ").strip().capitalize()
        sleep = float(input("Sleep Hours: "))
        papers = float(input("Sample Question Papers Practiced: "))

        if extra not in ("Yes", "No"):
            raise ValueError("Extracurricular must be 'Yes' or 'No'")
        extra_bin = 1 if extra == "Yes" else 0

        x_new = np.array([hours_studied, prev_scores, extra_bin, sleep, papers])
        prediction = intercept + np.dot(betas, x_new)

        print(f"\n🎯 Predicted Performance Index: {prediction:.2f}")

    except Exception as e:
        print(f"❌ Error during prediction: {e}")


if __name__ == "__main__":
    analyze_dataset("Student_Performance.csv")
    main("Student_Performance.csv")
