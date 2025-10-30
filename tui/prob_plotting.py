import matplotlib.pyplot as plt


def plot_polygon(x_vals: list[float], y_vals: list[float], file_name: str):
    if len(x_vals) != len(y_vals):
        print("ERROR: x_vals: list[float] != y_vals: list[float]")
        return

    # Построение полигона частот
    plt.plot(x_vals, y_vals, marker="o", color="blue", label="Полигон")

    plt.legend()
    plt.grid(True)
    save_fig_to_pdf_and_png(file_name)
    plt.close()


def save_fig_to_pdf_and_png(file_path: str) -> None:
    plt.savefig(file_path)
    if file_path.endswith(".png"):
        file_path = file_path[: -len(".png")] + ".pdf"
    else:
        file_path = file_path[: -len(".pdf")] + ".png"
    plt.savefig(file_path)


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
