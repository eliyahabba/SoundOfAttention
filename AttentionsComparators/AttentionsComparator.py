from abc import ABC, abstractmethod


class AttentionsComparator(ABC):
    """
    Abstract class for comparing attention weights.
    """

    @abstractmethod
    def create_attention_matrices(self, text, audio=None):
        pass

    #
    # @abstractmethod
    # def generate_insights(self, correlation_matrix):
    #     pass

    # @abstractmethod
    def compare_attention_matrices(self, model1_attention_matrix, model2_attention_matrix):
        pass
