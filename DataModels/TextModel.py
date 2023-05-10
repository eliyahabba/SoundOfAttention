from dataclasses import dataclass

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from Common.Constants import Constants

TextConstants = Constants.TextConstants


@dataclass
class TextModel:
    """
        A class representing a natural language processing model (e.g., BERT)

        Parameters:
        model_name (str): The name of the Language model
    """
    model_name: str = TextConstants.BERT_BASE
    device: torch.device = torch.device('cpu')

    def __post_init__(self):
        """
        Initialize the LanguageModel model with the provided name
        """
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, output_attentions=True)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
