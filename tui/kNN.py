import numpy as np
from typing import List, Tuple, Dict


def predict_knn(
    x_input: np.ndarray,
    x_data: np.ndarray,
    y_data: np.ndarray,
    k_values: List[int],
    threshold: float = 0.5,
) -> Dict[int, str]:
    """
    Perform k-Nearest Neighbors prediction for multiple k values.

    Parameters:
    -----------
    x_input : np.ndarray
        Input feature vector (1D array) of shape (n_features,)
    x_data : np.ndarray
        Training data matrix of shape (n_samples, n_features)
    y_data : np.ndarray
        Training labels (continuous or binary) of shape (n_samples,)
    k_values : List[int]
        List of k values to evaluate
    threshold : float, optional
        Threshold to convert continuous prediction to binary class (default: 0.5)

    Returns:
    --------
    Dict[int, str]
        Mapping from k -> formatted prediction string
    """
    if x_input.shape[0] != x_data.shape[1]:
        raise ValueError("Input feature dimension mismatch with training data.")

    distances = np.linalg.norm(x_data - x_input, axis=1)
    results = {}

    for k in k_values:
        if k > len(y_data):
            results[k] = f"Predicted Diabetes (k={k}): — (insufficient data)"
        else:
            nearest_idx = np.argsort(distances)[:k]
            pred = float(np.mean(y_data[nearest_idx]))  # Ensure float for JSON/etc.
            outcome = "yes" if pred >= threshold else "no"
            results[k] = f"🎯 Predicted Diabetes (k={k}): {pred:.2f} -> {outcome}"

    return results
