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


def plot_function(nums: list[float], func, file_path: str):
    x_values = nums.copy()
    x_values.append(min(nums) - 0.5)
    x_values.append(max(nums) + 0.5)
    x_values.sort()

    y = [func(x) for x in x_values]

    plt.step(x_values, y)
    plt.ylabel("F^{*}(x)")
    plt.title("Эмпирическая функция распределения")
    plt.grid(True)
    save_fig_to_pdf_and_png(file_path)
    plt.close()


def boxplot_for_two_lists(list1: List[float], list2: List[float], file_path: str):
    # Создаем график с двумя subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Боксплот для первого набора данных
    ax1.boxplot(list1, vert=True, patch_artist=True)
    ax1.set_title("Первый набор данных")
    ax1.set_ylabel("Значения")
    ax1.grid(True)

    ax2.boxplot(list2, vert=True, patch_artist=True)
    ax2.set_title("Второй набор данных")
    ax2.grid(True)

    plt.suptitle("Сравнение распределений данных")
    plt.tight_layout()

    save_fig_to_pdf_and_png(file_path)
    plt.close()


def save_fig_to_pdf_and_png(file_path: str) -> None:
    plt.savefig(file_path)
    if file_path.endswith(".png"):
        file_path = file_path[: -len(".png")] + ".pdf"
    else:
        file_path = file_path[: -len(".pdf")] + ".png"
    plt.savefig(file_path)
