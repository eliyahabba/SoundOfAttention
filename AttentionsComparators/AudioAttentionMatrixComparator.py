from typing import Tuple

import torch
from datasets import load_dataset

from AttentionExtractors.AudioAttentionExtractor import AudioAttentionExtractor
from AttentionsComparators.AttentionsComparator import AttentionsComparator
from Common.Constants import Constants
from Common.Utils.ProcessAudioData import ProcessAudioData
from DataModels.Attentions import Attentions
from CorrelationAnalysis import CorrelationAnalysis
AudioModelProcessorConstants = Constants.AudioModelProcessorConstants


class AudioAttentionMatrixComparator(AttentionsComparator):
    def __init__(self, audio_model_name1: str, audio_model_name2: str, device: torch.device,
                 correlation_analysis: CorrelationAnalysis):
        super().__init__(correlation_analysis)
        # Load the BERT model and tokenizer
        self.audio_model_name1 = audio_model_name1
        self.audio_model_name2 = audio_model_name2
        self.device = device
        self.audio_attention_extractor_model1 = AudioAttentionExtractor(audio_model_name1)
        self.audio_attention_extractor_model2 = AudioAttentionExtractor(audio_model_name2)

    def create_attention_matrices(self, audio: dict, audio_key: str = AudioModelProcessorConstants.LIBRISPEECH_AUDIO_KEY) -> Tuple[Attentions, Attentions]:
        audio_values = ProcessAudioData.get_audio_values(audio, audio_key)
        model1_attentions = self.audio_attention_extractor_model1.extract_attention(audio_values)
        model2_attentions = self.audio_attention_extractor_model2.extract_attention(audio_values)
        return model1_attentions, model2_attentions

    def predict_attentions_correlation(self, audio: dict, display_stats=False):
        model1_attentions, model2_attentions = self.create_attention_matrices(audio)
        assert model1_attentions.shape == model2_attentions.shape, \
            "The attention matrices should have the same shape"

        correlation_df = self.compare_attention_matrices(model1_attentions, model2_attentions)
        if display_stats:
            self.display_correlation_stats(audio, self.audio_model_name1, self.audio_model_name2, correlation_df)
        return correlation_df


if __name__ == '__main__':
    dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", 'clean', split='validation')

    # Access the loaded examples
    audio = dataset[0]["audio"]
    text = dataset[0]["text"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_attention_matrix_comparator = AudioAttentionMatrixComparator(
        audio_model_name1="facebook/wav2vec2-base-960h",
        audio_model_name2="facebook/wav2vec2-base-960h",
        device=device, correlation_analysis=CorrelationAnalysis())

    correlation_df = audio_attention_matrix_comparator.predict_attentions_correlation(audio, display_stats=True)
