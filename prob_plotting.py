from typing import List
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
