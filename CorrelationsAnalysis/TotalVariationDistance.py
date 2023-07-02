from abc import abstractmethod

import numpy as np

from CorrelationsAnalysis.CorrelationCalculatorInterface import CorrelationCalculatorInterface


class TotalVariationDistance(CorrelationCalculatorInterface):
    @staticmethod
    @abstractmethod
    def calculate_correlation(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """
        Calculate the Total Variation Distance between two attention matrices.

        Args:
            matrix1: First attention matrix as a numpy array.
            matrix2: Second attention matrix as a numpy array.

        Returns:
            Total Variation Distance between the two matrices as a float.
            Range: [0, 1]
            Lower values indicate higher similarity, and higher values indicate higher dissimilarity.
        """
        row_distances = []
        for row1, row2 in zip(matrix1, matrix2):
            # Calculate the Total Variation Distance between the two distributions
            tv_distance = 0.5 * np.sum(np.abs(row1 - row2))
            row_distances.append(tv_distance)

        # Average the distances across all rows
        tv_distance = np.mean(row_distances)

        return float(tv_distance)
