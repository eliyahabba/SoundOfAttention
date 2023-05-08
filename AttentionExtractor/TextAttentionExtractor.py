import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from AttentionExtractor.AttentionExtractor import AttentionExtractor


class TextAttentionExtractor(AttentionExtractor):
    def __init__(self, model_name: str):
        # use super() to call the parent class constructor
        super().__init__(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, output_attentions=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def run_model(self, text):
        # Tokenize the input text with both models
        tokens = self.tokenizer.tokenize(text)

        # TODO: check if we need to add special tokens for BERT
        # Add special tokens for BERT
        # bert_tokens = ['[CLS]'] + bert_tokens + ['[SEP]']

        # Convert the tokens to input IDs for both models
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Convert the input IDs to PyTorch tensors
        input_ids = torch.tensor(input_ids).unsqueeze(0)

        # Get the outputs and attention matrices for both models
        outputs = self.model(input_ids)
        return outputs

    def extract_attention(self, text, head):
        outputs = self.run_model(text)
        attentions = self.get_attention_matrix(outputs, head)
        return attentions
