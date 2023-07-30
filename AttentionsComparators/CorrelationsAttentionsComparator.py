import pandas as pd

from Common.Constants import Constants
from CorrelationsAnalysis.CorrelationAnalysis import CorrelationAnalysis
from DataModels.Attentions import Attentions
from DataModels.CorrelationsAttentionsComparisons import CorrelationsAttentionsComparisons

TextConstants = Constants.TextConstants
TextAttentionExtractorConstants = Constants.TextAttentionExtractorConstants
AttentionsConstants = Constants.AttentionsConstants


class CorrelationsAttentionsComparator:
    """
     Class for comparing attention weights.
    """

    def __init__(self, correlation_analysis: CorrelationAnalysis):
        self.correlation_analysis = correlation_analysis

    def compare_attention_matrices(self, attention_matrices1: Attentions,
                                   attention_matrices2: Attentions) -> CorrelationsAttentionsComparisons:
        """
        Calculate correlation between all matrices in arr_matrices1 to all matrices in arr_matrices2.

        :param attention_matrices1: First array of matrices. Shape [L, H, tokens, tokens]
        :param attention_matrices2: Second array of matrices. Shape [L,H, tokens, tokens]

        :return: Array of correlation between all matrices in arr_matrices1 to all matrices in arr_matrices2.
        Shape: [L,H,L,H]. Meaning Data[0,0,1,1] is the correlation between arr_matrices1[0,0] to arr_matrices2[1,1]
        """
        L = attention_matrices1.shape[AttentionsConstants.LAYER_AXIS]
        H = attention_matrices2.shape[AttentionsConstants.HEAD_AXIS]
        correlations_attentions_comparisons = CorrelationsAttentionsComparisons(layers_model_1=L, heads_model_1=H,
                                                                                layers_model_2=L, heads_model_2=H)
        if self.correlation_analysis.metric in ['jaccard', 'jaccard_T']:
            for l1 in range(L):
                for h1 in range(H):
                    matrix1 = attention_matrices1[l1][h1]
                    matrix2 = attention_matrices2[l1][h1]

                    attention_matrices1[l1][h1] = matrix1 > matrix1.mean() + 2 * matrix1.std()
                    attention_matrices2[l1][h1] = matrix2 > matrix2.mean() + 2 * matrix2.std()

        for l1 in range(L):
            for h1 in range(H):
                for l2 in range(L):
                    for h2 in range(H):
                        correlation = self.correlation_analysis.forward(attention_matrices1[l1][h1],
                                                                        attention_matrices2[l2][h2])
                        correlations_attentions_comparisons.set(l1, h1, l2, h2, correlation)

        return correlations_attentions_comparisons

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
