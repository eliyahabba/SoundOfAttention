from dataclasses import dataclass

from transformers import AutoModelForMaskedLM, AutoTokenizer

from Common.Constants import Constants
from DataModels.Model import Model

TextConstants = Constants.TextConstants


@dataclass
class TextModel(Model):

    def __post_init__(self):
        """
        Initialize the LanguageModel model with the provided name
        """
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_metadata.model_name, output_attentions=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_metadata.model_name)
