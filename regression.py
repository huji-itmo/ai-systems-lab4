import numpy as np
from typing import Union, List, Tuple


def multiple_linear_regression_scalar(
    y: Union[List[float], np.ndarray], X: Union[List[List[float]], np.ndarray]
) -> np.ndarray:
    """
    Выполняет множественную линейную регрессию методом нормальных уравнений
    (скалярный подход: построение и решение системы уравнений).

    Параметры
    ----------
    y : array-like, shape (n,)
        Зависимая переменная (вектор наблюдений).
    X : array-like, shape (n, p)
        Матрица независимых переменных (факторов), без столбца единиц.

    Возвращает
    ----------
    coeffs : np.ndarray, shape (p + 1,)
        Коэффициенты регрессии: [a, b1, b2, ..., bp],
        где `a` — свободный член, `b1..bp` — коэффициенты при факторах.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    if y.ndim != 1:
        raise ValueError("Вектор y должен быть одномерным.")
    if X.ndim != 2:
        raise ValueError("Матрица X должна быть двумерной.")
    if y.shape[0] != X.shape[0]:
        raise ValueError("Количество наблюдений в y и X должно совпадать.")

    n, p = X.shape

    # Добавляем столбец единиц для свободного члена
    X_with_const = np.column_stack([np.ones(n), X])  # shape (n, p+1)

    # Строим систему нормальных уравнений: (X'X) * coeffs = X'y
    XtX = X_with_const.T @ X_with_const  # (p+1) x (p+1)
    Xty = X_with_const.T @ y  # (p+1,)

    # Решаем систему
    coeffs = np.linalg.solve(XtX, Xty)
    return coeffs
