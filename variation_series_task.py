# from latex_generator import *
# from nums import get_first_task_nums
# from prob_plotting import plot_function, plot_polygon
# from variation_series import *


# def make_task1(precision):
#     nums = get_first_task_nums()

#     print(f"n = {len(nums)}")

#     print("Вариационный ряд:", get_variation_series_pretty(nums))
#     print(get_extremes(nums))
#     print("Размах: {:.2f}".format(get_whole_range(nums)))

#     statistical_series = get_statistical_series(nums)
#     path_to_statistical_series = "tex/output/task1/statistical_series.tex"
#     with open(path_to_statistical_series, "w") as f:
#         output = dict_to_latex_table_str(statistical_series)
#         f.write(output)
#     print(f"Статистический ряд: записаны в {path_to_statistical_series}")

#     plot_polygon(
#         statistical_series.keys(),
#         statistical_series.values(),
#         "tex/output/task1/polygon.pdf",
#     )

#     cumulative_values = []
#     prev = 0
#     for num in statistical_series.values():
#         cumulative_values.append(prev)
#         prev += num

#     plot_polygon(
#         statistical_series.keys(), cumulative_values, "tex/output/task1/cumulative.pdf"
#     )

#     ### оценки математического ожидания и среднеквадратического отклонения
#     expected_value = get_expected_value_estimate(nums)
#     confidence_prob = 0.9

#     print(
#         "Оценка математического ожидания (начальный момент): "
#         + str(round(expected_value, precision))
#     )

#     print(
#         "Стандартное отклонение"
#         + str(round(get_expected_value_deviation(nums), precision))
#     )
#     print(
#         "Дисперсия " + str(round(get_sample_variance(nums, expected_value), precision))
#     )

#     print(
#         "Оценка выборочного среднеквадратического отклонения: "
#         + str(round(get_sample_standard_deviation(nums, expected_value), precision))
#     )

#     print(
#         "Оценка выборочного с.к.о. (исправленная): "
#         + str(
#             round(
#                 get_sample_standard_deviation_corrected(nums, expected_value), precision
#             )
#         )
#     )

#     ### эмпирическая функция распределения и её график

#     plot_function(
#         nums,
#         lambda x: empirical_distribution_function(x, nums),
#         "tex/output/task1/distribution_function.pdf",
#     )
#     ### Помощь для теха
#     path_to_analytical_func_tex = "tex/output/task1/analytical_func.tex"
#     with open(path_to_analytical_func_tex, "w") as f:
#         f.write(
#             get_latex_form_empirical_distribution_func(
#                 nums, lambda x: empirical_distribution_function(x, nums)
#             )
#         )
#     print(
#         f"Вид эмпирической функции распределения: записан в {path_to_analytical_func_tex}"
#     )

#     for i in range(1, 4 + 1):
#         print(
#             f"Центральный момент {i} порядка: "
#             + str(round(moment_of_nth_order(nums, expected_value, i), precision))
#         )

#     print(f"Мода: {get_mode(nums)}")
#     print(f"Медиана: {get_median(nums)}")

#     print(
#         f"Коэффициент асимметрии: "
#         + str(round(moment_of_nth_order(nums, expected_value, 3), precision))
#     )

#     print(
#         f"Эксцесса: "
#         + str(round(moment_of_nth_order(nums, expected_value, 4) - 3, precision))
#     )


# # Выводы
