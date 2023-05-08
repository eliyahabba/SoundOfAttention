from dataclasses import dataclass

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

    def __post_init__(self):
        """
        Initialize the LanguageModel model with the provided name
        """
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, output_attentions=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
