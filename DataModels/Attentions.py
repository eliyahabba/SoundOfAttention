from dataclasses import dataclass

import numpy as np


@dataclass
class Attentions:
    attentions: np.ndarray

    def __getitem__(self, index: int):
        if isinstance(index, tuple):
            # if the index is a tuple, then it is a 2D index
            attention_layer, head = index
            return self.attentions[attention_layer][head]
        else:
            return self.attentions[index]

    # define shape attribute for the class
    @property
    def shape(self):
        return self.attentions.shape


