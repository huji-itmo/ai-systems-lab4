import numpy as np
import pandas as pd
from typing import Union, List


def multiple_linear_regression_scalar(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Solve multiple linear regression using the scalar method (normal equations).

    Parameters:
        y : np.ndarray, shape (n,)
            Dependent variable (target).
        X : np.ndarray, shape (n, p)
            Independent variables (features), without intercept column.

    Returns:
        coeffs : np.ndarray, shape (p + 1,)
            Regression coefficients [intercept, b1, b2, ..., bp].
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of samples in X and y must match.")

    # Add intercept column (ones)
    X_with_const = np.column_stack([np.ones(X.shape[0]), X])

    # Normal equations: (X'X) β = X'y
    XtX = X_with_const.T @ X_with_const
    Xty = X_with_const.T @ y

    # Solve for coefficients
    coeffs = np.linalg.solve(XtX, Xty)
    return coeffs


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
    y = df[target_column].values

    # Fit model
    coeffs = multiple_linear_regression_scalar(y, X)
    intercept = coeffs[0]
    betas = coeffs[1:]

    print("✅ Model trained successfully!")
    print(f"Intercept (a): {intercept:.4f}")
    for name, beta in zip(feature_columns, betas):
        print(f"Coefficient for '{name}': {beta:.4f}")

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
    # You can change this path if your file has a different name
    main("Student_Performance.csv")
