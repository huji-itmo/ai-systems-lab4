from interval_series import (
    get_interval_boarders,
    get_whole_range,
    get_interval_statistical_series,
)
from latex_generator import compute_table_to_latex_table_str_to_file
from variation_series import (
    get_expected_value_estimate,
    get_sample_standard_deviation_corrected,
    laplace_normalized_function,
)
from math import inf
import math


def calculate_chi_squared_statistic(
    data: list[float], num_intervals: int, precision: int
) -> float:
    """
    Performs Pearson's chi-squared test for normality assumption.

    Args:
        data: Input sample data
        num_intervals: Number of intervals for frequency analysis
        precision: Number of decimal places for rounding

    Returns:
        Calculated chi-squared statistic
    """
    # Data characteristics calculation
    data_range = get_whole_range(data)
    sample_mean = round(get_expected_value_estimate(data), precision)
    sample_std = get_sample_standard_deviation_corrected(data, sample_mean)

    # Interval calculations
    observed_frequencies = get_interval_statistical_series(
        data_range, num_intervals, data
    )
    interval_borders = get_interval_boarders(data_range, num_intervals, data)

    # Z-score calculations for interval borders
    z_scores = calculate_z_scores(
        interval_borders, sample_mean, sample_std, num_intervals, precision
    )

    # Generate interval analysis tables
    intervals_table = create_intervals_table(
        interval_borders, z_scores, sample_mean, num_intervals, precision
    )
    export_table_to_latex(intervals_table, "tex/output/hypotheses/intervals_table.tex")

    # Probability and frequency calculations
    expected_frequencies, probabilities = calculate_expected_frequencies(
        z_scores, num_intervals, len(data), precision
    )
    probabilities_table = create_probabilities_table(
        z_scores, probabilities, expected_frequencies, precision
    )
    export_table_to_latex(
        probabilities_table, "tex/output/hypotheses/probabilities_table.tex"
    )

    # Chi-squared statistic calculation
    chi_squared_components = calculate_chi_squared_components(
        observed_frequencies, expected_frequencies, precision
    )
    export_table_to_latex(
        chi_squared_components, "tex/output/hypotheses/chi_squared_components.tex"
    )

    return sum(row["component"] for row in chi_squared_components.values())


def calculate_z_scores(
    interval_borders: list[float],
    mean: float,
    std: float,
    num_intervals: int,
    precision: int,
) -> dict[int, float]:
    """Calculates z-scores for interval borders with edge handling."""
    z_scores = {}
    # Handle middle intervals
    for i in range(2, num_intervals):
        z_scores[i] = round((interval_borders[i] - mean) / std, precision)

    # Set edge values
    z_scores[1] = -inf
    z_scores[num_intervals] = inf
    return z_scores


def create_intervals_table(
    interval_borders: list[float],
    z_scores: dict[int, float],
    mean: float,
    num_intervals: int,
    precision: int,
) -> dict[int, dict]:
    """Creates table showing interval transformations."""
    table = {}
    for interval in range(1, num_intervals):
        table_entry = {
            "interval": (interval_borders[interval], interval_borders[interval + 1]),
            "centered_interval": (
                round(interval_borders[interval] - mean, precision),
                round(interval_borders[interval + 1] - mean, precision),
            ),
            "z_scores": (z_scores[interval], z_scores[interval + 1]),
        }
        table[interval] = table_entry
    return table


def calculate_expected_frequencies(
    z_scores: dict[int, float], num_intervals: int, sample_size: int, precision: int
) -> tuple[dict[int, float], dict[int, float]]:
    """Calculates expected frequencies and probabilities for intervals."""
    expected_frequencies = {}
    probabilities = {}

    for interval in range(1, num_intervals):
        phi_low = laplace_normalized_function(z_scores[interval])
        phi_high = laplace_normalized_function(z_scores[interval + 1])

        probability = phi_high - phi_low
        expected_freq = sample_size * probability

        probabilities[interval] = round(probability, precision)
        expected_frequencies[interval] = round(expected_freq, precision)

    return expected_frequencies, probabilities


def create_probabilities_table(
    z_scores: dict[int, float],
    probabilities: dict[int, float],
    expected_frequencies: dict[int, float],
    precision: int,
) -> dict[int, dict]:
    """Creates table showing probability calculations."""
    table = {}
    for interval in probabilities:
        table_entry = {
            "z_interval": (z_scores[interval], z_scores[interval + 1]),
            "phi_low": round(
                laplace_normalized_function(z_scores[interval]), precision
            ),
            "phi_high": round(
                laplace_normalized_function(z_scores[interval + 1]), precision
            ),
            "probability": probabilities[interval],
            "expected_frequency": expected_frequencies[interval],
        }
        table[interval] = table_entry
    return table


def calculate_chi_squared_components(
    observed: list[float], expected: dict[int, float], precision: int
) -> dict[int, dict]:
    """Calculates components for chi-squared statistic."""
    components = {}
    for idx, observed_freq in enumerate(observed, start=1):
        expected_freq = expected.get(idx, 0)
        difference = observed_freq - expected_freq
        component = (difference**2) / expected_freq if expected_freq != 0 else 0

        components[idx] = {
            "observed": round(observed_freq, precision),
            "expected": round(expected_freq, precision),
            "difference": round(difference, precision),
            "squared_diff": round(difference**2, precision),
            "component": round(component, precision),
        }
    return components


def export_table_to_latex(table: dict, filepath: str) -> None:
    """Exports analysis table to LaTeX format."""
    compute_table_to_latex_table_str_to_file(table, filepath)


# Normal distribution probability density function (not used in current calculation)
def normal_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-(x**2) / 2)
