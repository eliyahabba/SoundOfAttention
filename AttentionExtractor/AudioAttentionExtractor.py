from AttentionExtractor.AttentionExtractor import AttentionExtractor
from DataModels.Attentions import Attentions
from DataModels.AudioModel import AudioModel


class AudioAttentionExtractor(AttentionExtractor):
    """
    This class is responsible for extracting the attention matrices from the audio model.
    """

    def __init__(self, model_name: str):
        # use super() to call the parent class constructor
        super().__init__(model_name)
        self.audio_model = AudioModel(model_name)

    def run_model(self, text):
        # use the audio model to run the model
        pass

    def extract_attention(self, text)-> Attentions:
        outputs = self.run_model(text)
        attentions = self.get_attention_matrix(model_outputs=outputs)
        return attentions
