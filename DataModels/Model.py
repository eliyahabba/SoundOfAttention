from dataclasses import dataclass
from typing import Union

import torch
from transformers.modeling_outputs import MaskedLMOutput, Wav2Vec2BaseModelOutput

from DataModels.ModelMetadata import ModelMetadata


@dataclass
class Model:
    """
        A class representing an audio speech recognition model (e.g., Wav2Vec2)

        Parameters:
        model_name (str): The name of the Audio model
        data_type (str): The type of data the model is trained on (e.g., audio, text)
        align_tokens_to_bert_tokens (bool): Whether to align the tokens to the BERT tokens
        device (torch.device): The device to run the model on
    """
    model_metadata: ModelMetadata
    device: torch.device = torch.device('cpu')
    # The model itself (
    model: torch.nn.Module = None

    def __call__(self, input_ids: torch.Tensor) -> Union[MaskedLMOutput, Wav2Vec2BaseModelOutput]:
        # we don't train the model, so we set it to evaluation mode
        self.model.eval()
        # Move the model to the provided device
        self.model.to(self.device)
        # Use torch.inference_mode() to disable gradient computation and save memory
        with torch.inference_mode():
            return self.model(input_ids)
