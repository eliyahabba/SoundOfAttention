from abc import abstractmethod

import numpy as np
import torch

from DataModels.Attentions import Attentions


class AttentionExtractor:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.type = ''

    def get_attention_weights(self, model_outputs) -> np.ndarray:
        attentions = model_outputs.attentions
        # Concat the tensors across all heads
        attentions = torch.cat(attentions, dim=0)
        # Convert the PyTorch tensor to a NumPy array
        attentions = attentions.detach().numpy()
        return attentions

    def get_attention_matrix(self, model_outputs):
        attentions_outputs = self.get_attention_weights(model_outputs)
        attentions = Attentions(attentions_outputs)
        return attentions

    @abstractmethod
    def extract_attention(self, sample: dict, audio_key) -> Attentions:
        pass

    @abstractmethod
    def align_attentions(self, sample, attention) -> Attentions:
        pass
