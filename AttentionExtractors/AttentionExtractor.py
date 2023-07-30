from abc import abstractmethod

import numpy as np
import torch

from DataModels.Attentions import Attentions
from DataModels.ModelMetadata import ModelMetadata
from DataModels.Sample import Sample


class AttentionExtractor:
    def __init__(self, model_metadata: ModelMetadata):
        """
        This class is responsible for extracting the attention matrices from the model.
        :param model_metadata: model metadata object (contains the model name, data type, and if to align the tokens
        to the BERT tokens)
        """
        self.model_metadata = model_metadata

    def get_attention_weights(self, model_outputs) -> np.ndarray:
        attentions = model_outputs.attentions
        # Concat the tensors across all heads
        attentions = torch.cat(attentions, dim=0)
        # Convert the PyTorch tensor to a NumPy array
        if attentions.is_cuda:
            attentions = attentions.cpu()
        attentions = attentions.numpy()
        return attentions

    def get_attention_matrix(self, model_outputs) -> Attentions:
        """
        This method is responsible for extracting the attention matrices from the model.
        :param model_outputs: model outputs object (contains the attentions and the hidden states)
        :return: Attentions object
        """
        attentions_outputs = self.get_attention_weights(model_outputs)
        attentions = Attentions(attentions_outputs)
        return attentions

    @abstractmethod
    def extract_attention(self, sample: Sample) -> Attentions:
        """
        This method is responsible for extracting the attention matrices from the model.
        :param sample: sample from the dataset in a dictionary format
        :return: Attentions object
        """
        pass

    @abstractmethod
    def align_attentions(self, sample: Sample, attention: Attentions, use_cls_and_sep: bool) -> Attentions:
        pass
