# analyze_dataset.py (or keep in same file if preferred)

import os
from matplotlib import pyplot as plt
import pandas as pd
from prob_plotting import plot_polygon, save_fig_to_pdf_and_png
from variation_series import StatisticsAnalyzer


def plot_polygon_from_data(data: list[float], file_name: str):
    """Plot frequency polygon from raw data."""
    from collections import Counter

    counts = Counter(data)
    x_vals = sorted(counts.keys())
    y_vals = [float(counts[x]) for x in x_vals]
    plot_polygon(x_vals, y_vals, file_name)


def plot_empirical_cdf(data: list[float], file_path: str):
    """Plot empirical cumulative distribution function."""
    sorted_data = sorted(data)
    n = len(sorted_data)

    def empirical_cdf(x):
        return sum(1 for val in sorted_data if val <= x) / n

    # To make a smooth step plot, use all unique points + boundaries
    x_vals = [min(sorted_data) - 0.5] + sorted_data + [max(sorted_data) + 0.5]
    y_vals = [empirical_cdf(x) for x in x_vals]

    plt.step(x_vals, y_vals, where="post")
    plt.ylabel("F*(x)")
    plt.title("Эмпирическая функция распределения")
    plt.grid(True)
    save_fig_to_pdf_and_png(file_path)
    plt.close()


def boxplot_single(data: list[float], file_path: str):
    """Plot a single boxplot."""
    plt.figure(figsize=(6, 5))
    plt.boxplot(data, vert=True, patch_artist=True)
    plt.title("Boxplot")
    plt.ylabel("Значения")
    plt.grid(True)
    save_fig_to_pdf_and_png(file_path)
    plt.close()


def analyze_dataset(csv_path: str = "Student_Performance.csv", plot_dir: str = "plots"):
    os.makedirs(plot_dir, exist_ok=True)
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
            data_list = clean_data.tolist()
            analyzer = StatisticsAnalyzer(data_list)
            stats = {
                "count": len(clean_data),
                "mean": analyzer.get_expected_value_estimate(),
                "median": analyzer.get_median(),
                "mode": analyzer.get_mode(),
                "std": analyzer.get_sample_standard_deviation(),
                "std_corrected": analyzer.get_sample_standard_deviation_corrected(),
                "range": analyzer.get_whole_range(),
                "min": analyzer.get_extremes()[0],
                "max": analyzer.get_extremes()[1],
            }
            results[column] = stats

            # safe_name = "".join(c if c.isalnum() else "_" for c in column)
            # base_path = os.path.join(plot_dir, safe_name)

            # plot_polygon_from_data(data_list, f"{base_path}_polygon.png")
            # plot_empirical_cdf(data_list, f"{base_path}_ecdf.png")
            # boxplot_single(data_list, f"{base_path}_boxplot.png")

        except Exception as e:
            results[column] = {"error": str(e)}

    return results
