import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from AttentionExtractors.TextAttentionExtractor import TextAttentionExtractor
from AttentionsComparators.AttentionsComparator import AttentionsComparator
from Common.Constants import Constants

TextConstants = Constants.TextConstants
TextAttentionExtractorConstants = Constants.TextAttentionExtractorConstants
AttentionsConstants = Constants.AttentionsConstants


class TextAttentionMatrixComparator(AttentionsComparator):
    def __init__(self, text_model_name1: str, text_model_name2: str, device: torch.device):
        # Load the BERT model and tokenizer
        self.text_model_name1 = text_model_name1
        self.text_model_name2 = text_model_name2
        self.device = device
        self.text_attention_extractor_model1 = TextAttentionExtractor(text_model_name1, device)
        self.text_attention_extractor_model2 = TextAttentionExtractor(text_model_name2, device)

    def create_attention_matrices(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        model1_attentions = self.text_attention_extractor_model1.extract_attention(text)
        model2_attentions = self.text_attention_extractor_model2.extract_attention(text)
        return model1_attentions, model2_attentions

    def predict_attentions_correlation(self, text: str, diagonal_randomization, display_stats=False) -> pd.DataFrame:
        model1_attentions, model2_attentions = self.create_attention_matrices(text)
        assert model1_attentions.shape == model2_attentions.shape, \
            "The attention matrices should have the same shape"

        correlation_df = self.compare_attention_matrices(model1_attentions, model2_attentions, diagonal_randomization)
        if display_stats:
            self.display_correlation_stats(text, self.text_model_name1, self.text_model_name2, correlation_df)
        return correlation_df


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_name1', type=str, default=TextConstants.BERT_BASE,
                            help='The name of the first model to compare')
    arg_parser.add_argument('--model_name2', type=str, default=TextConstants.ROBERTA_BASE,
                            help='The name of the second model to compare')
    arg_parser.add_argument('--text', type=str, default='Hello, my dog is cute',
                            help='The text to extract the attention from')
    arg_parser.add_argument('--display_stats', type=bool, default=True,
                            help='Whether to display the correlation stats')
    arg_parser.add_argument('--diagonal_randomization', type=bool, default=False,
                            help='Whether to randomize the diagonal of the attention matrix, to avoid the correlation taken into account the diagonal')

    args = arg_parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_attention_matrix_comparator = TextAttentionMatrixComparator(text_model_name1=args.model_name1,
                                                                     text_model_name2=args.model_name2, device=device)
    correlation_df = text_attention_matrix_comparator.predict_attentions_correlation(args.text,
                                                                                     diagonal_randomization=args.diagonal_randomization,
                                                                                     display_stats=args.display_stats)
