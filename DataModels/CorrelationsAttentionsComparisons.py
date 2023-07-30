from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class CorrelationsAttentionsComparisons:
    """
    A class representing the correlations between attentions of two models.
    """

    def __init__(self, layers_model_1: int, heads_model_1: int, layers_model_2: int, heads_model_2: int) -> None:
        """
        Initialize the CorrelationsAttentionsComparisons object.
        :param layers_model_1: The number of layers in the first model.
        :param heads_model_1: The number of heads in the first model.
        :param layers_model_2: The number of layers in the second model.
        :param heads_model_2: The number of heads in the second model.
        """
        self.layers_model_1 = layers_model_1
        self.heads_model_1 = heads_model_1
        self.layers_model_2 = layers_model_2
        self.heads_model_2 = heads_model_2

        self.full_correlations_comparisons = np.zeros((layers_model_1, heads_model_1, layers_model_2, heads_model_2))

    def get_full_correlations_comparisons(self) -> np.ndarray:
        """
        Get the full correlations comparisons matrix.
        :return: The correlations comparisons matrix.
        """
        return self.full_correlations_comparisons

    def set(self, layer_model_1: int, head_model_1: int, layer_model_2: int, head_model_2: int, value: float) -> None:
        """
        Set the value of the correlation between the attentions of the two models.
        :param layer_model_1: The layer of the first model.
        :param head_model_1: The head of the first model.
        :param layer_model_2: The layer of the second model.
        :param head_model_2: The head of the second model.
        :param value: The value of the correlation.
        :return: None
        """
        self.full_correlations_comparisons[layer_model_1, head_model_1, layer_model_2, head_model_2] = value

    def get(self, layer_model_1: Union[int, None], head_model_1: Union[int, None], layer_model_2: Union[int, None],
            head_model_2: Union[int, None] = None) -> Union[np.ndarray, float]:
        """
        Get the value of the correlation between the attentions of the two models.
        :param layer_model_1: The layer of the first model.
        :param head_model_1: The head of the first model.
        :param layer_model_2: The layer of the second model.
        :param head_model_2: The head of the second model.
        :return: The value of the correlation.
        """
        return self.full_correlations_comparisons[layer_model_1, head_model_1, layer_model_2, head_model_2]
