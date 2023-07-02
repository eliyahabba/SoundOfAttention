from abc import abstractmethod

import numpy as np
from scipy.stats import pearsonr
from CorrelationsAnalysis.CorrelationCalculatorInterface import CorrelationCalculatorInterface


class PearsonCorrelation(CorrelationCalculatorInterface):
    @staticmethod
    @abstractmethod
    def calculate_correlation(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """
        Calculate the Pearson correlation coefficient between two matrices.

        Args:
            matrix1: First matrix as a numpy array.
            matrix2: Second matrix as a numpy array.

        Returns:
            Pearson correlation coefficient between the two matrices as a float.
            Range: [-1, 1]
            Higher values indicate higher similarity, and lower values indicate higher dissimilarity.
        """
        num_rows = matrix1.shape[0]
        correlations = []
        for i in range(num_rows):
            row1 = matrix1[i]
            row2 = matrix2[i]
            correlation, _ = pearsonr(row1, row2)
            correlations.append(correlation)

        mean_correlation = np.mean(correlations)
        return float(mean_correlation)
