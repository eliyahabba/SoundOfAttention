from typing import Tuple

import numpy as np


class CorrelationAnalysis:
    def __init__(self):
        pass

    @staticmethod
    def calculate_correlation(model1_attention_matrix: np.ndarray, model2_attention_matrix: np.ndarray,
                              diagonal_randomization=False) -> float:
        # if random_the_diagonal is True, then randomize the diagonal of the attention matrix
        if diagonal_randomization:
            model1_attention_matrix, model2_attention_matrix = CorrelationAnalysis.randomize_diagonals_values(
                model1_attention_matrix, model2_attention_matrix)
        # Calculate the correlation between the two matrices
        correlation_matrix = np.corrcoef(model1_attention_matrix, model2_attention_matrix)
        # Get the correlation between the two matrices
        correlation = correlation_matrix[0, 1]
        # Normalize the correlation
        normalized_correlation = (correlation + 1) / 2
        # truncate the correlation to 2 decimal places
        normalized_correlation = np.trunc(normalized_correlation * 100) / 100
        return normalized_correlation

    @staticmethod
    def randomize_diagonal_values(attention_matrix: np.ndarray) -> np.ndarray:
        # get the min, max and mean of each matrix
        min_attention_matrix = np.min(attention_matrix)
        max_attention_matrix = np.max(attention_matrix)
        # use the min, max and mean to generate a random number in the same distribution as the matrix
        random_model = np.random.uniform(min_attention_matrix, max_attention_matrix, attention_matrix.shape)
        # insert the random number in the diagonal of the attention matrix
        np.fill_diagonal(attention_matrix, random_model)
        return attention_matrix

    @staticmethod
    def randomize_diagonals_values(model1_attention_matrix: np.ndarray, model2_attention_matrix: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray]:
        model1_attention_matrix = CorrelationAnalysis.randomize_diagonal_values(model1_attention_matrix)
        model2_attention_matrix = CorrelationAnalysis.randomize_diagonal_values(model2_attention_matrix)
        return model1_attention_matrix, model2_attention_matrix
