from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class CorrelationsAttentionsComparisons:
    def __init__(self, layers_model_1, heads_model_1, layers_model_2, heads_model_2):
        self.layers_model_1 = layers_model_1
        self.heads_model_1 = heads_model_1
        self.layers_model_2 = layers_model_2
        self.heads_model_2 = heads_model_2

        self.full_correlations_comparisons = np.zeros((layers_model_1, heads_model_1, layers_model_2, heads_model_2))

    def get_full_correlations_comparisons(self):
        return self.full_correlations_comparisons

    def set(self, layer_model_1: int, head_model_1: int, layer_model_2: int, head_model_2: int, value: float):
        self.full_correlations_comparisons[layer_model_1, head_model_1, layer_model_2, head_model_2] = value

    def get(self, layer_model_1: Union[int, None], head_model_1: Union[int, None], layer_model_2: Union[int, None], head_model_2: Union[int, None] = None):
        return self.full_correlations_comparisons[layer_model_1, head_model_1, layer_model_2, head_model_2]
