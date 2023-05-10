import pandas as pd
import torch

from AttentionExtractor.AudioAttentionExtractor import AudioAttentionExtractor
from AttentionExtractor.TextAttentionExtractor import TextAttentionExtractor
from AttentionsComparators.AttentionsComparator import AttentionsComparator


class AudioTextAttentionMatrixComparator(AttentionsComparator):
    def __init__(self, text_model_name: str, audio_model_name: str, device: torch.device):
        self.text_model_name = text_model_name
        self.audio_model_name = audio_model_name
        self.device = device
        self.text_attention_extractor_model = TextAttentionExtractor(self.text_model_name)
        self.audio_attention_extractor_model = AudioAttentionExtractor(self.audio_model_name)

    def create_attention_matrices(self, audio, text):
        model1_attention = self.text_attention_extractor_model.extract_attention(text)
        model2_attention = self.audio_attention_extractor_model.extract_attention(audio)
        return model1_attention, model2_attention

    def predict_attentions_correlation(self, audio: pd.Series, display_stats=False):
        model1_attentions, model2_attentions = self.create_attention_matrices(audio)
        assert model1_attentions.shape == model2_attentions.shape, \
            "The attention matrices should have the same shape"

        correlation_df = self.compare_attention_matrices(model1_attentions, model2_attentions)
        if display_stats:
            self.display_correlation_stats(text_or_audio=audio, model_name1=self.text_model_name,
                                           model_name2=self.audio_model_name, correlation_df=correlation_df)
        return correlation_df


if __name__ == '__main__':
    pass
