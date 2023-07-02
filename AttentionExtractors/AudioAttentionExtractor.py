from AttentionExtractors.AttentionExtractor import AttentionExtractor
from ForcedAlignment.AudioTextAttentionsMatcher import AudioTextAttentionsMatcher
from Common.Utils.ProcessAudioData import ProcessAudioData
from DataModels.Attentions import Attentions
from DataModels.AudioModel import AudioModel
from DataModels.Sample import Sample
from Processors.AudioModelProcessor import AudioModelProcessor


class AudioAttentionExtractor(AttentionExtractor):
    """
    This class is responsible for extracting the attention matrices from the audio model.
    """

    def __init__(self, audio_model: AudioModel):
        # use super() to call the parent class constructor
        super().__init__(audio_model.model_metadata)
        self.audio_model_processor = AudioModelProcessor(audio_model)
        self.sample_type = 'audio'

    def extract_attention(self, sample: Sample) -> Attentions:
        audio_values = ProcessAudioData.get_audio_values(sample.audio)
        outputs = self.audio_model_processor.run(audio_values)  # Wav2Vec2BaseModelOutput
        attentions = self.get_attention_matrix(outputs)  # Attentions
        if self.model_metadata.align_tokens_to_bert_tokens:
            attentions = self.align_attentions(sample, attentions)
        return attentions

    def align_attentions(self, sample: Sample, attention):
        # group the audio_attention matrix by the matches
        aligned_audio_attention = AudioTextAttentionsMatcher.align_attentions(sample.audio, sample.text,
                                                                              attention)
        return aligned_audio_attention
