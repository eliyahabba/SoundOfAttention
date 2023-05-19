from typing import Tuple

import pandas as pd
import torch
from datasets import load_dataset

from AttentionExtractors.AudioAttentionExtractor import AudioAttentionExtractor
from AttentionExtractors.TextAttentionExtractor import TextAttentionExtractor
from AttentionsComparators.AttentionsComparator import AttentionsComparator
from AudioTextAttentionsMatcher import AudioTextAttentionsMatcher
from Common.Constants import Constants
from Common.Utils.ProcessAudioData import ProcessAudioData
from DataModels.Attentions import Attentions

AudioModelProcessorConstants = Constants.AudioModelProcessorConstants


class AudioTextAttentionMatrixComparator(AttentionsComparator):
    def __init__(self, text_model_name: str, audio_model_name: str, device: torch.device) -> None:
        self.text_model_name = text_model_name
        self.audio_model_name = audio_model_name
        self.device = device
        self.text_attention_extractor_model = TextAttentionExtractor(self.text_model_name)
        self.audio_attention_extractor_model = AudioAttentionExtractor(self.audio_model_name)

    def align_attentions(self, audio: dict, text: str,
                         audio_key: str = AudioModelProcessorConstants.LIBRISPEECH_AUDIO_KEY) -> Attentions:
        audio_values = ProcessAudioData.get_audio_values(audio, audio_key)
        audio_attention = self.audio_attention_extractor_model.extract_attention(audio_values)

        # group the audio_attention matrix by the matches
        aligned_audio_attention = AudioTextAttentionsMatcher.align_attentions(audio, text, audio_attention)

        return aligned_audio_attention

    def create_attention_matrices(self, audio: dict, text: str) -> Tuple[Attentions, Attentions]:
        text_attention = self.text_attention_extractor_model.extract_attention(text)
        aligned_audio_attention = self.align_attentions(audio, text)
        return text_attention, aligned_audio_attention

    def predict_attentions_correlation(self, audio: dict, text: str, display_stats=False)-> pd.DataFrame:
        model1_attentions, model2_attentions = self.create_attention_matrices(audio, text)
        assert model1_attentions.shape == model2_attentions.shape, \
            "The attention matrices should have the same shape"

        correlation_df = self.compare_attention_matrices(model1_attentions, model2_attentions)
        if display_stats:
            self.display_correlation_stats(text_or_audio=audio, model_name1=self.text_model_name,
                                           model_name2=self.audio_model_name, correlation_df=correlation_df)
        return correlation_df


if __name__ == '__main__':
    dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", 'clean', split='validation')
    audio = dataset[0]["audio"]
    text = dataset[0]["text"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_text_attention_matrix_comparator = AudioTextAttentionMatrixComparator(text_model_name="bert-base-uncased",
                                                                                audio_model_name="facebook/wav2vec2-base-960h",
                                                                                device=device)

    correlation_df = audio_text_attention_matrix_comparator.predict_attentions_correlation(audio, text,
                                                                                           display_stats=True)
