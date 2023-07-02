from abc import abstractmethod

import numpy as np
from scipy.stats import entropy

from CorrelationsAnalysis.CorrelationCalculatorInterface import CorrelationCalculatorInterface


class JensenShannonDivergence(CorrelationCalculatorInterface):
    @staticmethod
    @abstractmethod
    def calculate_correlation(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """
        Calculate the Jensen-Shannon Divergence between two attention matrices.

        Args:
            matrix1: First attention matrix as a numpy array.
            matrix2: Second attention matrix as a numpy array.

        Returns:
            Jensen-Shannon Divergence between the two matrices as a float.
            Range: (-inf, 0]
            Higher values indicate higher dissimilarity, and lower values indicate higher similarity.
        """
        row_divergences = []
        for row1, row2 in zip(matrix1, matrix2):
            # Compute the average of the two distributions
            avg_distribution = 0.5 * (row1 + row2)
            # Calculate the Jensen-Shannon Divergence using the average distribution
            js_divergence = 0.5 * (entropy(row1, avg_distribution) + entropy(row2, avg_distribution))
            row_divergences.append(js_divergence)

        # Average the divergences across all rows
        jensen_shannon_div = np.mean(row_divergences).item()

        return float(-jensen_shannon_div)
