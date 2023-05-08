from abc import ABC, abstractmethod


class AttentionsComparator(ABC):
    """
    Abstract class for comparing attention weights.
    """

    # @abstractmethod
    # def run_models(self, text, audio=None):
    #     """
    #     Run the models to get the attention weights.
    #     """
    #     pass
    #
    # @abstractmethod
    # def generate_insights(self, correlation_matrix):
    #     pass

    def compare_attention_matrices(self, model1_attention_matrix, model2_attention_matrix):
        pass
