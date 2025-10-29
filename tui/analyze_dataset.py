# analyze_dataset.py (or keep in same file if preferred)

import pandas as pd
from variation_series import StatisticsAnalyzer


def analyze_dataset(csv_path: str = "Student_Performance.csv"):
    df = pd.read_csv(csv_path)

    if "Extracurricular Activities" in df.columns:
        df["Extracurricular Activities"] = df["Extracurricular Activities"].map(
            {"Yes": 1.0, "No": 0.0}
        )

    numeric_df = df.select_dtypes(include=["number"])
    results = {}

    for column in numeric_df.columns:
        clean_data = numeric_df[column].dropna()
        if clean_data.empty:
            results[column] = {"error": "No valid data."}
            continue

        try:
            analyzer = StatisticsAnalyzer(clean_data.tolist())
            results[column] = {
                "count": len(clean_data),
                "mean": analyzer.get_expected_value_estimate(),
                "median": analyzer.get_median(),
                "mode": analyzer.get_mode(),
                "std_pop": analyzer.get_sample_standard_deviation(),
                "std_sample": analyzer.get_sample_standard_deviation_corrected(),
                "range": analyzer.get_whole_range(),
                "min": analyzer.get_extremes()[0],
                "max": analyzer.get_extremes()[1],
            }
        except Exception as e:
            results[column] = {"error": str(e)}

    return results
