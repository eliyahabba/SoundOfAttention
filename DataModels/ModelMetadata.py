from dataclasses import dataclass

from DataModels.DataType import DataType


@dataclass
class ModelMetadata:
    """
    This class is responsible for holding the metadata of a model.

    model_name: The name of the model
    data_type: The type of data the model is trained on (e.g., audio, text)
    align_to_text_tokens: Whether to align the tokens to the BERT tokens
    """

    model_name: str
    data_type: DataType
    align_to_text_tokens: bool = False
    aligner_tokenizer_name: str = 'bert-base-uncased'
    use_cls_and_sep: bool = False
