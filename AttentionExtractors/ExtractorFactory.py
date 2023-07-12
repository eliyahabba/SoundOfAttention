import torch

from AttentionExtractors.AttentionExtractor import AttentionExtractor
from AttentionExtractors.AudioAttentionExtractor import AudioAttentionExtractor
from AttentionExtractors.TextAttentionExtractor import TextAttentionExtractor
from DataModels.AudioModel import AudioModel
from DataModels.DataType import DataType
from DataModels.ModelMetadata import ModelMetadata
from DataModels.TextModel import TextModel


class ExtractorFactory:
    """
    This class is responsible for creating the attention extractor based on the type.
    """

    @staticmethod
    def create_attention_extractor(model_metadata: ModelMetadata,
                                   device: torch.device = torch.device('cpu')) -> AttentionExtractor:
        if model_metadata.data_type is DataType.Text:
            # convert model from Model to TextModel (TextModel inherits from Model)
            text_model = TextModel(model_metadata, device=device)
            return TextAttentionExtractor(text_model)
        elif model_metadata.data_type is DataType.Audio:
            # convert model from Model to AudioModel (AudioModel inherits from Model)
            audio_model = AudioModel(model_metadata, device=device)
            return AudioAttentionExtractor(audio_model)
        else:
            ValueError(f"Please provide 'type' that in {DataType.__members__.values()}")
