from abc import abstractmethod
import numpy as np
from CorrelationsAnalysis.CorrelationCalculatorInterface import CorrelationCalculatorInterface

class JaccardSimCoeff(CorrelationCalculatorInterface):
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
        matrix1 = matrix1 > matrix1.mean() + 2 * matrix1.std()
        matrix2 = matrix2 > matrix2.mean() + 2 * matrix2.std()

        matrix1 = matrix1[1:-1, 1:-1]
        matrix2 = matrix2[1:-1, 1:-1]

        i_1,j_1 = np.where(matrix1 > 0)
        indecis_mat_1 = set([(a,b) for a,b in zip(i_1, j_1)])

        i_2,j_2 = np.where(matrix2 > 0)
        indecis_mat_2 = set([(a,b) for a,b in zip(i_2, j_2)])

        if len(indecis_mat_2.intersection(indecis_mat_1)) == 0:
            return 0.0

        Jaccard_sim_coeff = len(indecis_mat_2.intersection(indecis_mat_1)) / len(indecis_mat_1.union(indecis_mat_2))
        return float(Jaccard_sim_coeff)
