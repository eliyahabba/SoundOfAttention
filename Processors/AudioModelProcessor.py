import numpy as np
import torch
from transformers.modeling_outputs import Wav2Vec2BaseModelOutput

from Common.Constants import Constants
from DataModels.AudioModel import AudioModel

AudioConstants = Constants.AudioConstants


class AudioModelProcessor:
    """
        A class representing an audio processing model (e.g., Wav2Vec2)

        Parameters:
            model_name (str): name of the model
            device (torch.device): device to run the model on
    """

    def __init__(self, model_name: str, device: torch.device = torch.device('cpu')):
        self.audio_model = AudioModel(model_name, device)

    def run(self, audio_values: np.ndarray) -> Wav2Vec2BaseModelOutput:
        inputs = self.audio_model.processor(audio_values, sampling_rate=16_000, return_tensors="pt", padding=True)
        inputs = inputs.to(self.audio_model.device)
        outputs = self.audio_model.model(inputs.input_values)
        return outputs
