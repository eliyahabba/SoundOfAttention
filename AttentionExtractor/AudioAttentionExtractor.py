import numpy as np
import torch

from AttentionExtractor.AttentionExtractor import AttentionExtractor
from DataModels.Attentions import Attentions
from Processors.AudioModelProcessor import AudioModelProcessor


class AudioAttentionExtractor(AttentionExtractor):
    """
    This class is responsible for extracting the attention matrices from the audio model.
    """

    def __init__(self, model_name: str, device: torch.device = torch.device('cpu')):
        # use super() to call the parent class constructor
        super().__init__(model_name)
        self.audio_model_processor = AudioModelProcessor(model_name, device)

    def extract_attention(self, audio_values: np.ndarray) -> Attentions:
        outputs = self.audio_model_processor.run(audio_values)  # Wav2Vec2BaseModelOutput
        attentions = self.get_attention_matrix(outputs)  # Attentions
        return attentions
