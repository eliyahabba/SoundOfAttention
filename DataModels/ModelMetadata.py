from dataclasses import dataclass

from DataModels.DataType import DataType


@dataclass
class ModelMetadata:
    """
    This class is responsible for holding the metadata of a model.

    model_name: The name of the model
    data_type: The type of data the model is trained on (e.g., audio, text)
    align_tokens_to_bert_tokens: Whether to align the tokens to the BERT tokens
    """

    model_name: str
    data_type: DataType
    align_tokens_to_bert_tokens: bool = False
