from typing import Union

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from Common.Constants import Constants
from CorrelationAnalysis import CorrelationAnalysis

AttentionsConstants = Constants.AttentionsConstants


class AttentionsComparator():
    """
    Abstract class for comparing attention weights.
    """

    def __init__(self, correlation_analysis: CorrelationAnalysis):
        self.correlation_analysis = correlation_analysis

    def compare_attention_matrices(self, model1_attention_matrix, model2_attention_matrix):
        results = []
        for layer in range(model1_attention_matrix.shape[AttentionsConstants.LAYER_AXIS]):
            for head in range(model1_attention_matrix.shape[AttentionsConstants.HEAD_AXIS]):
                correlation = self.correlation_analysis.forward(model1_attention_matrix[layer][head],
                                                                model2_attention_matrix[layer][head])

                # correlation = CorrelationAnalysis.calculate_correlation(model1_attention_matrix[layer][head],
                #                                                         model2_attention_matrix[layer][head],
                #                                                         diagonal_randomization=diagonal_randomization)
                results.append({AttentionsConstants.LAYER: layer, AttentionsConstants.HEAD: head,
                                AttentionsConstants.CORRELATION: correlation})
        results_df = pd.DataFrame(results)
        # convert the column to int type
        results_df[AttentionsConstants.LAYER] = results_df[AttentionsConstants.LAYER].astype(int)
        results_df[AttentionsConstants.HEAD] = results_df[AttentionsConstants.HEAD].astype(int)

        results_df.sort_values(by=[AttentionsConstants.CORRELATION], inplace=True, ascending=False)
        return results_df

    def print_correlation_result(self, text_or_audio: Union[str, pd.Series], model_name1: str, model_name2: str,
                                 correlation_df: pd.DataFrame, top_k=5):
        print(f'Comparing the attention matrices for the text: {text_or_audio}')
        print(f'Using the models: {model_name1} and {model_name2}')
        print(f'The best correlation is {correlation_df.iloc[0][AttentionsConstants.CORRELATION]:.2f} '
              f'between layer {correlation_df[AttentionsConstants.LAYER].values[0]} and head {correlation_df[AttentionsConstants.HEAD].values[0]}')
        print(f'The worst correlation is {correlation_df.iloc[-1][AttentionsConstants.CORRELATION]:.2f} '
              f'between layer {correlation_df[AttentionsConstants.LAYER].values[-1]} and head {correlation_df[AttentionsConstants.HEAD].values[-1]}')

        # print the top k results with the highest correlation without the index
        print(f'The top {top_k} results with the highest correlation are:')
        print(correlation_df.head(top_k).to_string(index=False))
        # print the top k results with the lowest correlation
        print(f'The top {top_k} results with the lowest correlation are:')
        print(correlation_df.tail(top_k).to_string(index=False))

    def display_correlation_stats(self, text_or_audio: Union[str, dict], model_name1: str, model_name2: str,
                                  correlation_df: pd.DataFrame, top_k=5):
        self.print_correlation_result(text_or_audio, model_name1, model_name2, correlation_df, top_k)
        self.plot_correlation_result_hitmap(text_or_audio, model_name1, model_name2, correlation_df)

    def plot_correlation_result_hitmap(self, text_or_audio: Union[str, pd.Series], model_name1: str, model_name2: str,
                                       correlation_df: pd.DataFrame):
        # Plot the correlation matrix
        fig, ax = plt.subplots(figsize=(10, 12))
        correlation_matrix = correlation_df.pivot(index='layer', columns='head', values=AttentionsConstants.CORRELATION)
        ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1, ax=ax,
                         annot_kws={"fontsize": 11, "fontweight": 'bold'})
        title = f'Correlation between the attention matrices for the text:\n ' \
                f'{text_or_audio}\n' \
                f'Using the models: 1.{model_name1}. 2.{model_name2}'
        # add the title (wrap it  it doesn't get cut off)
        ax.set_title(title, fontsize=14, fontweight='bold', wrap=True)

        ax.set_xlabel('Head')
        ax.set_ylabel('Layer')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        # Save the plot
        # ax.figure.savefig(f'../Results/Correlation/{model_name1}_{model_name2}_{text}.png')
        # Show the plot
        plt.show()
