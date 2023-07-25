import warnings

from AttentionExtractors.AttentionExtractor import AttentionExtractor
from DataModels import Attentions
from DataModels.Sample import Sample
from DataModels.TextModel import TextModel
from Processors.TextModelProcessor import TextModelProcessor


class TextAttentionExtractor(AttentionExtractor):
    """
    This class is responsible for extracting the attention matrices from the text model.
    """

    def __init__(self, text_model: TextModel):
        # use super() to call the parent class constructor
        super().__init__(text_model.model_metadata)
        self.text_model_processor = TextModelProcessor(text_model)

    def extract_attention(self, sample: Sample) -> Attentions:
        outputs = self.text_model_processor.run(sample.text)  # MaskedLMOutput
        attentions = self.get_attention_matrix(outputs)  # Attentions
        if self.model_metadata.align_to_text_tokens:
            attentions = self.align_attentions(sample, attentions,
                                               use_cls_and_sep=self.text_model_processor.text_model.model_metadata.use_cls_and_sep)
        return attentions

    def align_attentions(self, sample: Sample, attention: Attentions, use_cls_and_sep: bool):
        # Print a warning message
        warnings.warn("The align_attentions method is not implemented for the text model.")
        return attention
