from transformers import BertTokenizer, BertModel

from Common.Constants import Constants

TextConstants = Constants.TextConstants


class TextModel:
    """
        A class representing a natural language processing model (e.g., BERT)
    """

    def __init__(self, model_name: str = TextConstants.BERT):
        """
        Initialize the LanguageModel model with the provided name

        Parameters:
        model_name (str): The name of the Language model
        """
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)

    def generate_attention_matrix(self, text):
        """
        Get the attention layer for the provided text sample

        Parameters:
        text (str): The input text

        Returns:
        np.ndarray: A numpy array representing the attention layer
        """
        # Add code to process the text and return the attention layer
        pass
