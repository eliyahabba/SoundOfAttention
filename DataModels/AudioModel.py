from dataclasses import dataclass

from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model

from Common.Constants import Constants

AudioConstants = Constants.AudioConstants


@dataclass
class AudioModel:
    """
        A class representing an audio speech recognition model (e.g., Wav2Vec2)

        Parameters:
        model_name (str): The name of the Audio model
    """
    model_name: str = AudioConstants.W2V2

    def __post_init__(self):
        """
        Initialize the AudioModel model with the provided name
        """

        # Initialize the tokenizer and model
        tokenizer_class, model_class = self._get_audio_model_and_tokenizer(self.model_name)
        self.tokenizer = tokenizer_class.from_pretrained(self.model_name)
        self.model = model_class.from_pretrained(self.model_name, output_attentions=True)

    @staticmethod
    def _get_audio_model_and_tokenizer(model_name_or_path):
        """
        Get the audio model and tokenizer based on the provided model name or path
        :param model_name_or_path: The name or path of the audio model
        :return: The audio tokenizer and model classes
        """
        # Map the model name to the corresponding tokenizer and model class
        if model_name_or_path == AudioConstants.W2V2:
            tokenizer_class = Wav2Vec2Tokenizer
            model_class = Wav2Vec2Model
        else:
            raise ValueError("Unknown audio model:", model_name_or_path)
        return tokenizer_class, model_class
