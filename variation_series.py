from collections import OrderedDict
import math
import scipy
import numpy as np
from q_table import get_q_table_value

np.set_printoptions(legacy="1.25")


class StatisticsAnalyzer:
    def __init__(self, data: list[float]):
        if not data:
            raise ValueError("Input data list cannot be empty.")
        self.data = data
        self._sorted_data = None
        self._statistical_series = None

    @property
    def sorted_data(self) -> list[float]:
        if self._sorted_data is None:
            self._sorted_data = sorted(self.data)
        return self._sorted_data

    @property
    def statistical_series(self) -> OrderedDict[float, int]:
        if self._statistical_series is None:
            freq = {}
            for num in self.data:
                freq[num] = freq.get(num, 0) + 1
            self._statistical_series = OrderedDict(sorted(freq.items()))
        return self._statistical_series

    # --- Basic descriptive statistics ---

    def get_variation_series(self) -> list[float]:
        return self.sorted_data

    def get_variation_series_pretty(self) -> str:
        return " ≤ ".join(map(str, self.sorted_data))

    def get_whole_range(self) -> float:
        return self.sorted_data[-1] - self.sorted_data[0]

    def get_extremes(self) -> tuple[float, float]:
        return (self.sorted_data[0], self.sorted_data[1])

    def get_mode(self) -> float | None:
        max_freq = max(self.statistical_series.values())
        for key, freq in self.statistical_series.items():
            if freq == max_freq:
                return key
        return None

    def get_median(self) -> float:
        n = len(self.sorted_data)
        mid = n // 2
        if n % 2 == 0:
            return (self.sorted_data[mid - 1] + self.sorted_data[mid]) / 2
        else:
            return self.sorted_data[mid]

    # --- Expected value and deviation ---

    def get_expected_value_estimate(self) -> float:
        return sum(self.data) / len(self.data)

    def get_expected_value_deviation(self) -> float:
        expected = self.get_expected_value_estimate()
        return sum(x - expected for x in self.data)

    # --- Variance and standard deviation ---

    def get_sample_variance(self, expected_value: float | None = None) -> float:
        ev = (
            expected_value
            if expected_value is not None
            else self.get_expected_value_estimate()
        )
        return sum((x - ev) ** 2 for x in self.data) / len(self.data)

    def moment_of_nth_order(
        self, nth_order: int, expected_value: float | None = None
    ) -> float:
        ev = (
            expected_value
            if expected_value is not None
            else self.get_expected_value_estimate()
        )
        return sum((x - ev) ** nth_order for x in self.data) / len(self.data)

    def get_sample_standard_deviation(
        self, expected_value: float | None = None
    ) -> float:
        variance = self.get_sample_variance(expected_value)
        return math.sqrt(variance)

    def get_sample_standard_deviation_corrected(
        self, expected_value: float | None = None
    ) -> float:
        variance = self.get_sample_variance(expected_value)
        n = len(self.data)
        if n <= 1:
            raise ValueError("Cannot compute corrected standard deviation for n <= 1.")
        return math.sqrt(variance * (n / (n - 1)))

    # --- Distribution and confidence intervals ---

    def empirical_distribution_function(self, x: float) -> float:
        count = sum(1 for value in self.data if value < x)
        return count / len(self.data)

    @staticmethod
    def laplace_function(x: float) -> float:
        return scipy.stats.norm.cdf(x)

    @staticmethod
    def laplace_normalized_function(x: float) -> float:
        return StatisticsAnalyzer.laplace_function(x) - 0.5

    @staticmethod
    def student_coefficient(gamma: float, degrees_of_freedom: int) -> float:
        return scipy.stats.t.ppf(gamma, degrees_of_freedom)

    @staticmethod
    def get_inverse_laplace(alpha: float, error_margin: float = 0.0001) -> float:
        step = 2.0
        last_point = step
        while (
            abs(StatisticsAnalyzer.laplace_function(last_point) - alpha) > error_margin
        ):
            if StatisticsAnalyzer.laplace_function(last_point) < alpha:
                last_point += step
            else:
                last_point -= step
            step /= 2
        return last_point

    def get_confidence_interval_for_expvalue(
        self, confidence_prob: float, sigma: float | None = None
    ) -> tuple[float, float]:
        n = len(self.data)
        avg = self.get_expected_value_estimate()
        sigma_used = (
            sigma if sigma is not None else self.get_sample_standard_deviation()
        )

        if n <= 30:
            t_gamma = self.student_coefficient(confidence_prob / 2 + 0.5, n - 1)
        else:
            t_gamma = self.get_inverse_laplace(confidence_prob / 2 + 0.5)

        margin = t_gamma * sigma_used / math.sqrt(n)
        return avg - margin, avg + margin

    def get_confidence_interval_for_standard_deviation(
        self, gamma: float
    ) -> tuple[float, float]:
        n = len(self.data)
        sigma_corr = self.get_sample_standard_deviation_corrected()
        q = get_q_table_value(gamma, n)
        if q < 1:
            return sigma_corr * (1 - q), sigma_corr * (1 + q)
        else:
            return 0.0, sigma_corr * (1 + q)
