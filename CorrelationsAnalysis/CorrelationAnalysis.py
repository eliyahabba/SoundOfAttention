from typing import Tuple, Type

import numpy as np

from CorrelationsAnalysis.CorrelationCalculatorInterface import CorrelationCalculatorInterface
from CorrelationsAnalysis.CosineSimilarity import CosineSimilarity
from CorrelationsAnalysis.JensenShannonDivergence import JensenShannonDivergence
from CorrelationsAnalysis.KullbackLeiblerDivergence import KullbackLeiblerDivergence
from CorrelationsAnalysis.PearsonCorrelation import PearsonCorrelation
from CorrelationsAnalysis.TotalVariationDistance import TotalVariationDistance


class CorrelationAnalysis:
    def __init__(self, metric: str = 'Cosine',
                 diagonal_randomization=False):
        """
        Initialize the CorrelationAnalysis class.

        Args:
            metric: String indicating the correlation metric to use.
                    Possible values: 'KL', 'JS', 'Cosine', 'tot_var', 'pearson'
        """
        self.metric = metric
        self.correlation_method = self.get_correlation_method(metric)
        self.diagonal_randomization = diagonal_randomization

    @staticmethod
    def get_correlation_method(metric: str) -> Type[CorrelationCalculatorInterface]:
        """
        Return the appropriate correlation function based on the given metric.

        Args:
            metric: String indicating the correlation metric.

        Returns:
            The corresponding correlation function.
        """
        if metric == 'KL':
            return KullbackLeiblerDivergence
        elif metric == 'JS':
            return JensenShannonDivergence
        elif metric == 'Cosine':
            return CosineSimilarity
        elif metric == 'tot_var':
            return TotalVariationDistance
        elif metric == 'pearson':
            return PearsonCorrelation
        else:
            raise ValueError(
                "Invalid correlation metric. Supported options: 'KL', 'JS', 'Cosine', 'tot_var', 'pearson'")

    def preprocess(self, matrix1: np.ndarray, matrix2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        if self.diagonal_randomization:
            matrix1, matrix2 = CorrelationAnalysis.randomize_diagonals_values(matrix1, matrix2)

        return matrix1, matrix2

    def forward(self, matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """
        Compute the correlation measure between two matrices.

        Args:
            matrix1: First matrix as a numpy array.
            matrix2: Second matrix as a numpy array.

        Returns:
            Correlation measure between the two matrices as a float.
        """
        matrix1, matrix2 = self.preprocess(matrix1, matrix2)
        return self.correlation_method.calculate_correlation(matrix1, matrix2)

    def run_all(self, matrix1: np.ndarray, matrix2: np.ndarray) -> dict:
        """
        Compute all correlation measures between two matrices.

        Args:
            matrix1: First matrix as a numpy array.
            matrix2: Second matrix as a numpy array.

        Returns:
            A dictionary containing all correlation measures as key-value pairs.
        """

        matrix1, matrix2 = self.preprocess(matrix1, matrix2)
        results = {}
        correlation_metrics = ['KL', 'JS', 'Cosine', 'tot_var', 'pearson']
        for metric in correlation_metrics:
            correlation_method = self.get_correlation_method(metric)
            result = correlation_method.calculate_correlation(matrix1, matrix2)
            results[metric] = result
        return results

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
    def randomize_diagonals_values(matrix1: np.ndarray, matrix2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        matrix1 = CorrelationAnalysis.randomize_diagonal_values(matrix1)
        matrix2 = CorrelationAnalysis.randomize_diagonal_values(matrix2)
        return matrix1, matrix2
