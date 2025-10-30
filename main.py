import pandas as pd
import numpy as np
import sys


def predict_knn(
    x_input: np.ndarray,
    x_data: np.ndarray,
    y_data: np.ndarray,
    k_values: list,
    threshold: float = 0.5,
) -> dict:
    """Distance-weighted kNN prediction."""
    if x_input.shape[0] != x_data.shape[1]:
        raise ValueError("Input dimension mismatch with training data.")

    distances = np.linalg.norm(x_data - x_input, axis=1)
    eps = 1e-8
    results = {}

    for k in k_values:
        if k > len(y_data):
            results[k] = f"❌ k={k}: Not enough data (need at least {k} samples)"
            continue

        nearest_idx = np.argsort(distances)[:k]
        neighbor_distances = distances[nearest_idx]
        neighbor_labels = y_data[nearest_idx]

        weights = 1.0 / (neighbor_distances + eps)
        pred = float(np.average(neighbor_labels, weights=weights))
        outcome = "yes" if pred >= threshold else "no"
        results[k] = f"🎯 k={k}: score = {pred:.3f} → Diabetes = {outcome}"

    return results


def main():
    csv_path = "diabetes.csv"

    # Load data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(
            f"❌ Error: '{csv_path}' not found. Please ensure the file is in the current directory."
        )
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

    # Define features and target
    feature_names = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "Pedigree",
        "Age",
    ]
    target = "Outcome"

    # Validate columns
    missing_cols = [col for col in feature_names + [target] if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns in dataset: {missing_cols}")
        sys.exit(1)

    # Extract data
    X = df[feature_names].values.astype(float)
    y = df[target].values.astype(float)

    # Show dataset stats
    print("📊 Dataset Overview:")
    print(f"   Total samples: {len(df)}")
    print(f"   Features: {', '.join(feature_names)}")
    print(f"   Target: {target} (0 = No, 1 = Yes)")
    print("\n📈 Feature Statistics:")
    print("-" * 60)
    for col in feature_names:
        series = df[col].dropna()
        print(
            f"{col:>15}: mean={series.mean():6.2f}, std={series.std():6.2f}, "
            f"min={series.min():6.1f}, max={series.max():6.1f}"
        )
    print("-" * 60)
    print()

    # Get input from user
    print("🩺 Enter your values below (press Enter to skip and use a random patient):")
    user_values = {}
    has_input = False
    for feat in feature_names:
        while True:
            val = input(f"  {feat:>15}: ").strip()
            if val == "":
                user_values[feat] = None
                break
            try:
                num = float(val)
                if num < 0:
                    print("    ⚠️ Value must be non-negative. Try again.")
                    continue
                user_values[feat] = num
                has_input = True
                break
            except ValueError:
                print("    ⚠️ Invalid number. Try again.")

    # Use random row if no input
    if not has_input:
        print("\n🎲 No input provided. Using a random patient from the dataset...")
        random_row = df.sample(n=1).iloc[0]
        for feat in feature_names:
            user_values[feat] = float(random_row[feat])
    else:
        # Fill missing with random values
        for feat in feature_names:
            if user_values[feat] is None:
                user_values[feat] = float(df[feat].dropna().sample(n=1).iloc[0])
        print("\n📝 Missing values filled with random samples from the dataset.")

    # Prepare input vector
    x_input = np.array([user_values[feat] for feat in feature_names])

    # Run predictions
    print("\n🔍 Running kNN predictions (distance-weighted)...")
    k_values = [3, 5, 10]
    predictions = predict_knn(x_input, X, y, k_values)

    print("\n✅ Results:")
    print("-" * 50)
    for k in k_values:
        print(predictions[k])
    print("-" * 50)
    print("\n💡 Note: Prediction ≥ 0.5 → 'yes' (diabetes likely)")


if __name__ == "__main__":
    main()
