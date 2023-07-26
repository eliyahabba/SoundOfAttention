from AttentionExtractors.AttentionExtractor import AttentionExtractor
from Common.Utils.ProcessAudioData import ProcessAudioData
from DataModels.Attentions import Attentions
from DataModels.AudioModel import AudioModel
from DataModels.Sample import Sample
from ForcedAlignment.AudioTextAttentionsMatcher import AudioTextAttentionsMatcher
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
        self.audio_text_attentions_matcher = AudioTextAttentionsMatcher()

    def extract_attention(self, sample: Sample) -> Attentions:
        audio_values = ProcessAudioData.get_audio_values(sample.audio)
        outputs = self.audio_model_processor.run(audio_values)  # Wav2Vec2BaseModelOutput
        attentions = self.get_attention_matrix(outputs)  # Attentions
        if self.model_metadata.align_tokens_to_bert_tokens:
            attentions = self.align_attentions(sample, attentions,
                                               use_cls_and_sep=self.audio_model_processor.audio_model.model_metadata.use_cls_and_sep)
        return attentions

    def align_attentions(self, sample: Sample, attention: Attentions, use_cls_and_sep: bool):
        # group the audio_attention matrix by the matches
        aligned_audio_attention = self.audio_text_attentions_matcher.align_attentions(sample.audio, sample.text,
                                                                                      attention,
                                                                                      use_cls_and_sep=use_cls_and_sep)
        return aligned_audio_attention
