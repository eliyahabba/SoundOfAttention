import torch
from transformers.modeling_outputs import MaskedLMOutput

from Common.Constants import Constants
from DataModels.TextModel import TextModel

TextConstants = Constants.TextConstants


class TextModelProcessor:
    """
        A class representing a Text processing model (e.g., BERT)

        Parameters:
            model_name (str): name of the model
            device (torch.device): device to run the model on

    """

    def __init__(self, model_name: str, device: torch.device = torch.device('cpu')):
        self.text_model = TextModel(model_name, device)

    def tokenize_text(self, text: str) -> torch.Tensor:
        # Tokenize the input text with the text model's tokenizer
        tokens = self.text_model.tokenizer.tokenize(text)

        # TODO: check if we need to add special tokens for BERT
        # Add special tokens for BERT
        # bert_tokens = ['[CLS]'] + bert_tokens + ['[SEP]']

        # Convert the tokens to input IDs for the text model
        input_ids = self.text_model.tokenizer.convert_tokens_to_ids(tokens)

        # Convert the input IDs to PyTorch tensors
        input_ids = torch.tensor(input_ids).unsqueeze(0)

        # Move the input tensors to the device
        input_ids = input_ids.to(self.text_model.device)
        return input_ids

    def run(self, text: str) -> MaskedLMOutput:
        # Tokenize the text and get the input IDs
        input_ids = self.tokenize_text(text)

        # Get the outputs from the text model
        outputs = self.text_model.model(input_ids)
        return outputs
