import pandas as pd
import torch

from AttentionExtractor.AudioAttentionExtractor import AudioAttentionExtractor
from AttentionsComparators.AttentionsComparator import AttentionsComparator


class AudioTextAttentionMatrixComparator(AttentionsComparator):
    def __init__(self, audio_model_name1: str, audio_model_name2: str, device: torch.device):
        # Load the BERT model and tokenizer
        self.audio_model_name1 = audio_model_name1
        self.audio_model_name2 = audio_model_name2
        self.device = device
        self.audio_attention_extractor_model1 = AudioAttentionExtractor(audio_model_name1)
        self.audio_attention_extractor_model2 = AudioAttentionExtractor(audio_model_name2)

    def create_attention_matrices(self, audio):
        model1_attentions = self.audio_attention_extractor_model1.extract_attention(audio)
        model2_attentions = self.audio_attention_extractor_model2.extract_attention(audio)
        return model1_attentions, model2_attentions

    def predict_attentions_correlation(self, audio: pd.Series, display_stats=False):
        model1_attentions, model2_attentions = self.create_attention_matrices(audio)
        assert model1_attentions.shape == model2_attentions.shape, \
            "The attention matrices should have the same shape"

        correlation_df = self.compare_attention_matrices(model1_attentions, model2_attentions)
        if display_stats:
            self.display_correlation_stats(audio, self.audio_model_name1, self.audio_model_name2, correlation_df)
        return correlation_df


if __name__ == '__main__':
    pass
