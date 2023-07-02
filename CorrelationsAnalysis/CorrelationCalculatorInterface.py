from abc import abstractmethod

import numpy as np


class CorrelationCalculatorInterface:
    @staticmethod
    @abstractmethod
    def calculate_correlation(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """
        Calculate the correlation between two attention matrices.

        Args:
            matrix1: First attention matrix as a numpy array.
            matrix2: Second attention matrix as a numpy array.

        Returns:
            Correlation between the two matrices as a float.
            Range: [0, 1]
            Higher values indicate higher similarity, and lower values indicate higher dissimilarity.
        """
        raise NotImplementedError
