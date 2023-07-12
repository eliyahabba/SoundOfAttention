from abc import abstractmethod

import numpy as np
from scipy.stats import entropy

from CorrelationsAnalysis.CorrelationCalculatorInterface import CorrelationCalculatorInterface
from CorrelationsAnalysis.CorrelationCalculatorInterface import CorrelationCalculatorInterface


class KullbackLeiblerDivergence(CorrelationCalculatorInterface):
    @staticmethod
    @abstractmethod
    def calculate_correlation(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """
        Calculate the Kullback-Leibler (KL) Divergence between two attention matrices.

        Args:
            matrix1: First attention matrix as a numpy array.
            matrix2: Second attention matrix as a numpy array.

        Returns:
            Kullback-Leibler Divergence between the two matrices as a float.
            Range: (-inf, 0]
            Higher values indicate higher dissimilarity, and lower values indicate higher similarity.
        """
        row_divergences = []
        for row1, row2 in zip(matrix1, matrix2):
            # Calculate the Kullback-Leibler Divergence for the two distributions
            kl_divergence = entropy(row1, row2)
            row_divergences.append(kl_divergence)

        # Average the divergences across all rows
        kl_divergence = np.mean(row_divergences)

        return float(-kl_divergence)
