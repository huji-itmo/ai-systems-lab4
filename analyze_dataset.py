import pandas as pd
from variation_series import StatisticsAnalyzer


def analyze_dataset(csv_path: str = "Student_Performance.csv"):
    # Load data
    df = pd.read_csv(csv_path)

    # Optionally preprocess (e.g., convert "Yes"/"No" to 1/0)
    if "Extracurricular Activities" in df.columns:
        df["Extracurricular Activities"] = df["Extracurricular Activities"].map(
            {"Yes": 1.0, "No": 0.0}
        )

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        print("⚠️ No numeric columns found in the dataset.")
        return

    print("=" * 80)
    print(f"📊 STATISTICAL SUMMARY FOR EACH VARIABLE IN: {csv_path}")
    print("=" * 80)

    for column in numeric_df.columns:
        print(f"\n📈 VARIABLE: '{column}'")
        print("-" * 60)

        # Drop NaN values
        clean_data = numeric_df[column].dropna()
        if clean_data.empty:
            print("   ❌ No valid data.")
            continue

        try:
            analyzer = StatisticsAnalyzer(clean_data.tolist())

            mean = analyzer.get_expected_value_estimate()
            median = analyzer.get_median()
            mode = analyzer.get_mode()
            std = analyzer.get_sample_standard_deviation()
            std_corr = analyzer.get_sample_standard_deviation_corrected()
            data_range = analyzer.get_whole_range()
            extremes = analyzer.get_extremes()
            n = len(clean_data)

            print(f"   Count (n):          {n}")
            print(f"   Mean:               {mean:.4f}")
            print(f"   Median:             {median:.4f}")
            print(
                f"   Mode:               {mode:.4f}"
                if mode is not None
                else "   Mode:               None"
            )
            print(f"   Std (population):   {std:.4f}")
            print(f"   Std (sample, corr): {std_corr:.4f}")
            print(f"   Range:              {data_range:.4f}")
            print(f"   {extremes}")

        except Exception as e:
            print(f"   ❌ Error analyzing '{column}': {e}")

    print("\n" + "=" * 80)
    print("✅ Analysis complete.")


if __name__ == "__main__":
    analyze_dataset("Student_Performance.csv")
