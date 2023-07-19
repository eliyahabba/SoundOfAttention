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
        input_ids = self.text_model.tokenizer.encode(text,
                                                     add_special_tokens=self.text_model.model_metadata.use_cls_and_sep,
                                                     return_tensors='pt')

        # Move the input tensors to the device
        input_ids = input_ids.to(self.text_model.device)
        return input_ids

    def run(self, text: str) -> MaskedLMOutput:
        # Tokenize the text and get the input IDs
        input_ids = self.tokenize_text(text)

        # Get the outputs from the text model
        outputs = self.text_model(input_ids)
        return outputs
