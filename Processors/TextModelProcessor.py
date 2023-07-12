import torch
from transformers.modeling_outputs import MaskedLMOutput

from Common.Constants import Constants
from DataModels.TextModel import TextModel

TextConstants = Constants.TextConstants


class TextModelProcessor:
    """
        A class representing a Text processing model (e.g., BERT)

        Parameters:
            text_model (TextModel): The text model
    """

    def __init__(self, text_model: TextModel):
        self.text_model = text_model

    def tokenize_text(self, text: str) -> torch.Tensor:
        # Tokenize the input text with the text model's tokenizer

        # tokens = self.text_model.tokenizer.tokenize(text)
        input_ids = self.text_model.tokenizer(text, add_special_tokens=self.text_model.model_metadata.use_cls_and_sep)['input_ids']
        # tokens = self.text_model.tokenizer.convert_ids_to_tokens(input_ids)
        # TODO: check if we need to add special tokens for BERT
        # Add special tokens for BERT
        # encoded_input = self.text_model.tokenizer.encode_plus("[CLS] " + text + " [SEP]", add_special_tokens=True, return_tensors='pt')
        # # Convert the input IDs to PyTorch tensors
        # input_ids = encoded_input['input_ids']

        # Convert the tokens to input IDs for the text model
        # input_ids = self.text_model.tokenizer.convert_tokens_to_ids(tokens)

        # Convert the input IDs to PyTorch tensors
        input_ids = torch.tensor(input_ids).unsqueeze(0)

        # Move the input tensors to the device
        input_ids = input_ids.to(self.text_model.device)
        return input_ids

    def run(self, text: str) -> MaskedLMOutput:
        # Tokenize the text and get the input IDs
        input_ids = self.tokenize_text(text)

        # Get the outputs from the text model
        outputs = self.text_model(input_ids)
        return outputs
