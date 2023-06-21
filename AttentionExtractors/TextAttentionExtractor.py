import torch
from transformers.modeling_outputs import MaskedLMOutput

from .AttentionExtractor import AttentionExtractor
from DataModels import Attentions
from Processors.TextModelProcessor import TextModelProcessor


class TextAttentionExtractor(AttentionExtractor):
    """
    This class is responsible for extracting the attention matrices from the text model.
    """

    def __init__(self, model_name: str, device: torch.device = torch.device('cpu')):
        # use super() to call the parent class constructor
        super().__init__(model_name)
        self.text_model_processor = TextModelProcessor(model_name, device)

    def extract_attention(self, text) -> Attentions:
        outputs = self.text_model_processor.run(text)  # MaskedLMOutput
        attentions = self.get_attention_matrix(outputs)  # Attentions
        return attentions
