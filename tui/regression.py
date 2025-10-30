import numpy as np
from typing import Union, List


def multiple_linear_regression_scalar(
    y: Union[List[float], np.ndarray], X: Union[List[List[float]], np.ndarray]
) -> tuple[np.ndarray, np.float64, np.float64]:
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

    X_with_const = np.column_stack([np.ones(n), X])  # shape (n, p+1)

    XtX = X_with_const.T @ X_with_const  # (p+1) x (p+1)
    Xty = X_with_const.T @ y  # (p+1,)

    coeffs = np.linalg.solve(XtX, Xty)

    X_with_const = np.column_stack([np.ones(X.shape[0]), X])
    y_pred = X_with_const @ coeffs

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    r = np.corrcoef(y, y_pred)[0, 1]

    return coeffs, r, r_squared
