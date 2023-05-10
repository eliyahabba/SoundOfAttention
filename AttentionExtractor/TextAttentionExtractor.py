import torch
from transformers.modeling_outputs import MaskedLMOutput

from AttentionExtractor.AttentionExtractor import AttentionExtractor
from DataModels import Attentions
from DataModels.TextModel import TextModel


class TextAttentionExtractor(AttentionExtractor):
    """
    This class is responsible for extracting the attention matrices from the text model.
    """

    def __init__(self, model_name: str, device: torch.device = torch.device('cpu')):
        # use super() to call the parent class constructor
        super().__init__(model_name)
        self.text_model = TextModel(model_name, device)

    def run_model(self, text) -> MaskedLMOutput:
        # Tokenize the input text with both models
        tokens = self.text_model.tokenizer.tokenize(text)

        # TODO: check if we need to add special tokens for BERT
        # Add special tokens for BERT
        # bert_tokens = ['[CLS]'] + bert_tokens + ['[SEP]']

        # Convert the tokens to input IDs for both models
        input_ids = self.text_model.tokenizer.convert_tokens_to_ids(tokens)

        # Convert the input IDs to PyTorch tensors
        input_ids = torch.tensor(input_ids).unsqueeze(0)

        # Move the tensors to the device
        input_ids = input_ids.to(self.text_model.device)
        # Get the outputs and attention matrices for both models
        outputs = self.text_model.model(input_ids)
        return outputs

    def extract_attention(self, text) -> Attentions:
        outputs = self.run_model(text)
        attentions = self.get_attention_matrix(outputs)
        return attentions
