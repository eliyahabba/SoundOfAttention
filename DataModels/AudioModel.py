from dataclasses import dataclass

import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from Common.Constants import Constants

AudioConstants = Constants.AudioConstants


@dataclass
class AudioModel:
    """
        A class representing an audio speech recognition model (e.g., Wav2Vec2)

        Parameters:
        model_name (str): The name of the Audio model
    """
    model_name: str = AudioConstants.W2V2
    device: torch.device = torch.device('cpu')

    def __post_init__(self):
        """
        Initialize the AudioModel model with the provided name
        """

        # Initialize the tokenizer and model
        model_class, processor_class = self._get_audio_model_and_tokenizer(self.model_name)
        self.model = model_class.from_pretrained(self.model_name, output_attentions=True)
        self.processor = processor_class.from_pretrained(self.model_name)
        # Move the model to the provided device
        self.model.to(self.device)

    @staticmethod
    def _get_audio_model_and_tokenizer(model_name_or_path: str) -> (Wav2Vec2Model, Wav2Vec2Processor):
        """
        Get the audio model and tokenizer based on the provided model name or path
        :param model_name_or_path: The name or path of the audio model
        :return: The audio tokenizer and model classes
        """
        # Map the model name to the corresponding tokenizer and model class
        if model_name_or_path == AudioConstants.W2V2:
            model_class = Wav2Vec2Model
            processor_class = Wav2Vec2Processor
        else:
            raise ValueError("Unknown audio model:", model_name_or_path)
        return model_class, processor_class
