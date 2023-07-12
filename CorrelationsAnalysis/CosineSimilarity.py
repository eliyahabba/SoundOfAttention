from abc import abstractmethod

import numpy as np
from scipy.spatial.distance import cosine

from CorrelationsAnalysis.CorrelationCalculatorInterface import CorrelationCalculatorInterface


class CosineSimilarity(CorrelationCalculatorInterface):
    @staticmethod
    @abstractmethod
    def calculate_correlation(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """
        Calculate the cosine similarity between two attention matrices row by row.

        Args:
            matrix1: First attention matrix as a numpy array.
            matrix2: Second attention matrix as a numpy array.

        Returns:
            Cosine similarity between the two matrices as a float.
            Range: [-1, 1]
            Higher values indicate higher similarity, and lower values indicate higher dissimilarity.
        """
        num_rows = matrix1.shape[0]
        similarities = []
        for i in range(num_rows):
            row1 = matrix1[i]
            row2 = matrix2[i]
            similarity = 1 - cosine(row1, row2)
            similarities.append(similarity)

        average_similarity = np.mean(similarities)
        return float(average_similarity)

    @staticmethod
    def calculate_correlation_old(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """
        Calculate the cosine similarity between two matrices.

        Args:
            matrix1: First matrix as a numpy array.
            matrix2: Second matrix as a numpy array.

        Returns:
            Cosine similarity between the two matrices as a float.
            Range: [-1, 1]
            Higher values indicate higher similarity, and lower values indicate higher dissimilarity.
        """
        # Flatten the matrices into 1D arrays
        flattened1 = matrix1.flatten()
        flattened2 = matrix2.flatten()

        # Calculate the cosine similarity
        similarity = 1.0 - cosine(flattened1, flattened2)

        return similarity
