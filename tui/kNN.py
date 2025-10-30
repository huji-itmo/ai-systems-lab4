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
    Perform distance-weighted k-Nearest Neighbors prediction for multiple k values.

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

    # Small epsilon to prevent division by zero
    eps = 1e-8

    for k in k_values:
        if k > len(y_data):
            results[k] = f"Predicted Diabetes (k={k}): — (insufficient data)"
        else:
            # Get indices of k nearest neighbors
            nearest_idx = np.argsort(distances)[:k]
            neighbor_distances = distances[nearest_idx]
            neighbor_labels = y_data[nearest_idx]

            # Compute inverse-distance weights
            weights = 1.0 / (neighbor_distances + eps)

            # Weighted average prediction
            pred = float(np.average(neighbor_labels, weights=weights))

            outcome = "yes" if pred >= threshold else "no"
            results[k] = f"🎯 Predicted Diabetes (k={k}): {pred:.2f} -> {outcome}"

    return results
