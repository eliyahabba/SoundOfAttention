import numpy as np
import torch

from AttentionExtractors.AttentionExtractor import AttentionExtractor
from DataModels.Attentions import Attentions
from Processors.AudioModelProcessor import AudioModelProcessor
from AudioTextAttentionsMatcher import AudioTextAttentionsMatcher
from Common.Utils.ProcessAudioData import ProcessAudioData


class AudioAttentionExtractor(AttentionExtractor):
    """
    This class is responsible for extracting the attention matrices from the audio model.
    """

    def __init__(self, model_name: str, device: torch.device = torch.device('cpu')):
        # use super() to call the parent class constructor
        super().__init__(model_name)
        self.audio_model_processor = AudioModelProcessor(model_name, device)
        self.type = 'audio'

    def extract_attention(self, sample: dict, audio_key) -> Attentions:
        audio_values = ProcessAudioData.get_audio_values(sample['audio'], audio_key)
        outputs = self.audio_model_processor.run(audio_values)  # Wav2Vec2BaseModelOutput
        attentions = self.get_attention_matrix(outputs)  # Attentions
        return attentions

    def align_attentions(self, sample, attention):
        # group the audio_attention matrix by the matches
        aligned_audio_attention = AudioTextAttentionsMatcher.align_attentions(sample['audio'], sample['text'],
                                                                              attention)
        return aligned_audio_attention
