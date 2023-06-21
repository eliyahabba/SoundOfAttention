import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from typing import Tuple

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
        self.correlation_func = self.get_correlation_function(metric)
        self.diagonal_randomization = diagonal_randomization

    @staticmethod
    def get_correlation_function(metric: str):
        """
        Return the appropriate correlation function based on the given metric.

        Args:
            metric: String indicating the correlation metric.

        Returns:
            The corresponding correlation function.
        """
        if metric == 'KL':
            return CorrelationAnalysis.kullback_leibler_divergence
        elif metric == 'JS':
            return CorrelationAnalysis.jensen_shannon_divergence
        elif metric == 'Cosine':
            return CorrelationAnalysis.cosine_similarity
        elif metric == 'tot_var':
            return CorrelationAnalysis.total_variation_distance
        elif metric == 'pearson':
            return CorrelationAnalysis.pearson_correlation
        else:
            raise ValueError(
                "Invalid correlation metric. Supported options: 'KL', 'JS', 'Cosine', 'tot_var', 'pearson'")

    @staticmethod
    def jensen_shannon_divergence(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """
        Calculate the Jensen-Shannon Divergence between two attention matrices.

        Args:
            matrix1: First attention matrix as a numpy array.
            matrix2: Second attention matrix as a numpy array.

        Returns:
            Jensen-Shannon Divergence between the two matrices as a float.
            Range: [0, inf)
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

        return jensen_shannon_div

    @staticmethod
    def kullback_leibler_divergence(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """
        Calculate the Kullback-Leibler (KL) Divergence between two attention matrices.

        Args:
            matrix1: First attention matrix as a numpy array.
            matrix2: Second attention matrix as a numpy array.

        Returns:
            Kullback-Leibler Divergence between the two matrices as a float.
            Range: [0, inf)
            Higher values indicate higher dissimilarity, and lower values indicate higher similarity.
        """
        row_divergences = []
        for row1, row2 in zip(matrix1, matrix2):
            # Calculate the Kullback-Leibler Divergence for the two distributions
            kl_divergence = entropy(row1, row2)
            row_divergences.append(kl_divergence)

        # Average the divergences across all rows
        kl_divergence = np.mean(row_divergences)

        return float(kl_divergence)

    @staticmethod
    def total_variation_distance(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
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

    @staticmethod
    def cosine_similarity_old(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
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

    @staticmethod
    def cosine_similarity(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
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
    def pearson_correlation(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
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
        # Flatten the matrices into 1D arrays
        flattened1 = matrix1.flatten()
        flattened2 = matrix2.flatten()

        # Calculate the Pearson correlation coefficient
        correlation, _ = pearsonr(flattened1, flattened2)

        return correlation

    def preprocess(self, matrix1: np.ndarray, matrix2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

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
        return self.correlation_func(matrix1, matrix2)

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
            correlation_func = self.get_correlation_function(metric)
            result = correlation_func(matrix1, matrix2)
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
