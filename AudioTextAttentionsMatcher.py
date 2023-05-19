from typing import Dict

import numpy as np
from datasets import load_dataset

from Common.Resources import BasicResources
from Common.Utils.ProcessAudioData import ProcessAudioData
from DataModels.Attentions import Attentions
from ForcedAlignment.TextAudioMatcher import TextAudioMatcher


class AudioTextAttentionsMatcher:
    @staticmethod
    def align_text_audio(audio: dict, text: str):
        text_audio_matcher = TextAudioMatcher(BasicResources())
        matches = text_audio_matcher.match(text, audio)
        return matches

    @staticmethod
    def align_attentions(audio: Dict[str, any], text: str, audio_attention: Attentions) -> Attentions:
        matches = AudioTextAttentionsMatcher.align_text_audio(audio, text)

        # group the audio_attention matrix by the matches
        grouped_audio_attention = AudioTextAttentionsMatcher.group_attention_matrix_by_matches(audio_attention,
                                                                                               matches)

        return grouped_audio_attention

    @staticmethod
    def group_attention_matrix_by_matches(audio_attention, matches):
        aggregated_attention = np.zeros((audio_attention.attentions.shape[0], audio_attention.attentions.shape[1],
                                         len(matches), len(matches)))
        # Iterate over the index tuples
        for i, match_row in enumerate(matches):
            start_row, end_row = match_row['audio_start'], match_row['audio_end']
            for j, match_col in enumerate(matches):
                start_col, end_col = match_col['audio_start'], match_col['audio_end']
                # Extract the chunk from the matrix based on the index ranges
                chunk = audio_attention.attentions[:, :, start_row:end_row, start_col:end_col]

                # Perform aggregation on the chunk (e.g., sum, mean, max, etc.)
                aggregated_value = np.median(chunk, axis=(2, 3))
                aggregated_attention[:, :, i, j] = aggregated_value
        audio_attention.attentions = aggregated_attention
        return audio_attention


if __name__ == "__main__":
    dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", 'clean', split='validation')
    audio = dataset[0]["audio"]
    text = dataset[0]["text"]
    audio_model_name = "facebook/wav2vec2-base-960h"
    audio_key = "array"

    from AttentionExtractors.AudioAttentionExtractor import AudioAttentionExtractor

    audio_attention_extractor_model = AudioAttentionExtractor(audio_model_name)
    audio_values = ProcessAudioData.get_audio_values(audio, audio_key)
    audio_attention = audio_attention_extractor_model.extract_attention(audio_values)

    # group the audio_attention matrix by the matches
    aligned_audio_attention = AudioTextAttentionsMatcher.align_attentions(audio, text, audio_attention)
