import numpy as np
import pandas as pd

from Common.Constants import Constants
from CorrelationsAnalysis.CorrelationAnalysis import CorrelationAnalysis
from DataModels.Attentions import Attentions

TextConstants = Constants.TextConstants
TextAttentionExtractorConstants = Constants.TextAttentionExtractorConstants
AttentionsConstants = Constants.AttentionsConstants


class AttentionsComparator:
    """
     Class for comparing attention weights.
    """

    def __init__(self, correlation_analysis: CorrelationAnalysis):
        self.correlation_analysis = correlation_analysis

    def compare_attention_matrices(self, attention_matrices1: Attentions,
                                   attention_matrices2: Attentions) -> np.ndarray:
        """
        Calculate correlation between all matrices in arr_matrices1 to all matrices in arr_matrices2.

        :param attention_matrices1: First array of matrices. Shape [L, H, tokens, tokens]
        :param attention_matrices2: Second array of matrices. Shape [L,H, tokens, tokens]

        :return: Array of correlation between all matrices in arr_matrices1 to all matrices in arr_matrices2.
        Shape: [L,H,L,H]. Meaning Data[0,0,1,1] is the correlation between arr_matrices1[0,0] to arr_matrices2[1,1]
        """
        L = attention_matrices1.shape[AttentionsConstants.LAYER_AXIS]
        H = attention_matrices2.shape[AttentionsConstants.HEAD_AXIS]
        full_correlations_comparisons = np.zeros((L, H, L, H))

        for l1 in range(L):
            for h1 in range(H):
                for l2 in range(L):
                    for h2 in range(H):
                        full_correlations_comparisons[l1, h1, l2, h2] = self.correlation_analysis.forward(
                            attention_matrices1[l1][h1],
                            attention_matrices2[l2][h2])

        return full_correlations_comparisons

    def compare_head_to_head(self, attention_matrices1: Attentions, attention_matrices2: Attentions) -> pd.DataFrame:
        """
        Calculate correlation between all heads in arr_matrices1 to all heads in arr_matrices2.
        :param attention_matrices1: First array of matrices. Shape [L, H, tokens, tokens]
        :param attention_matrices2: Second array of matrices. Shape [L,H, tokens, tokens]
        :return: DataFrame of correlation between all heads in arr_matrices1 to all heads in arr_matrices2.
        """
        results = []
        for layer in range(attention_matrices1.shape[AttentionsConstants.LAYER_AXIS]):
            for head in range(attention_matrices1.shape[AttentionsConstants.HEAD_AXIS]):
                correlation = self.correlation_analysis.forward(attention_matrices1[layer][head],
                                                                attention_matrices2[layer][head])

                results.append({AttentionsConstants.LAYER: layer, AttentionsConstants.HEAD: head,
                                AttentionsConstants.CORRELATION: correlation})
        results_df = pd.DataFrame(results)
        # convert the column to int type
        results_df[AttentionsConstants.LAYER] = results_df[AttentionsConstants.LAYER].astype(int)
        results_df[AttentionsConstants.HEAD] = results_df[AttentionsConstants.HEAD].astype(int)

        results_df.sort_values(by=[AttentionsConstants.CORRELATION], inplace=True, ascending=False)
        return results_df
