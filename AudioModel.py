import numpy as np
from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model
from Constants import Constants
AudioConstants = Constants.AudioConstants

class AudioModel:
    """
        A class representing an audio speech recognition model (e.g., Wav2Vec2)
    """
    def __init__(self, model_name: str=AudioConstants.W2V2):
        """
        Initialize the AudioModel model with the provided name

        Parameters:
        model_name (str): The name of the Audio model
        """
        self.model_name = model_name
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(self.model_name)
        self.model = Wav2Vec2Model.from_pretrained(self.model_name)


    def generate_attention_matrix(self, audio)-> np.ndarray:
        """
        Get the attention layer for the provided audio sample

        Parameters:
        audio (np.ndarray): A numpy array representing the audio sample

        Returns:
        np.ndarray: A numpy array representing the attention layer
        """
        pass
