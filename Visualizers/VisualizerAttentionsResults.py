import numpy as np
import pandas as pd
# import seaborn as sns
from matplotlib import pyplot as plt

from Common.Constants import Constants
from DataModels.CorrelationsAttentionsComparisons import CorrelationsAttentionsComparisons
from DataModels.Sample import Sample

AttentionsConstants = Constants.AttentionsConstants


class VisualizerAttentionsResults:
    @staticmethod
    def plot_correlation_of_attentions(sample: Sample, model_name1: str,
                                       model_name2: str,
                                       correlations_attentions_comparisons: CorrelationsAttentionsComparisons):
        # Plot the correlation matrix
        full_correlations_comparisons = correlations_attentions_comparisons.get_full_correlations_comparisons()
        correlations_comparisons_flat = full_correlations_comparisons.reshape(
            full_correlations_comparisons.shape[0] * full_correlations_comparisons.shape[1],
            full_correlations_comparisons.shape[2] * full_correlations_comparisons.shape[3])
        correlation_df = pd.DataFrame(correlations_comparisons_flat)
        fig, ax = plt.subplots(figsize=(40, 40))
        ax = sns.heatmap(correlation_df, annot=True, cmap='coolwarm', vmin=0, vmax=1, ax=ax,
                         annot_kws={"fontsize": 11, "fontweight": 'bold'})

        title = f'Correlation between the full attention matrices for the text:\n ' \
                f'{sample.text}\n' \
                f'Using the models: 1.{model_name1}. 2.{model_name2}'
        # add the title (wrap it doesn't get cut off)
        ax.set_title(title, fontsize=14, fontweight='bold', wrap=True)

        ax.set_xlabel('All attentions Model 1')
        ax.set_ylabel('All attentions Model 2')
        # Show the plot
        plt.show()

    @staticmethod
    def plot_correlation_of_attentions_by_avg_of_each_layer(sample: Sample, model_name1: str,
                                                            model_name2: str,
                                                            correlations_attentions_comparisons: CorrelationsAttentionsComparisons):
        correlation_df = pd.DataFrame(correlations_attentions_comparisons)
        fig, ax = plt.subplots(figsize=(10, 12))
        ax = sns.heatmap(correlation_df, annot=True, cmap='coolwarm', vmin=0, vmax=1, ax=ax,
                         annot_kws={"fontsize": 11, "fontweight": 'bold'})
        title = f'Correlation between the attention matrices by averaging each layer for the text:\n ' \
                f'{sample.text}\n' \
                f'Using the models: 1.{model_name1}. 2.{model_name2}'
        # add the title (wrap it doesn't get cut off)
        ax.set_title(title, fontsize=14, fontweight='bold', wrap=True)

        ax.set_xlabel('Layers Model 1')
        ax.set_ylabel('Layers Model 2')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        # Show the plot
        plt.show()

    @staticmethod
    def plot_correlation_of_attentions_by_comparing_head_to_head(sample: Sample, model_name1: str, model_name2: str,
                                                                 correlation_df: pd.DataFrame):
        # Plot the correlation matrix
        fig, ax = plt.subplots(figsize=(10, 12))
        correlation_matrix = correlation_df.pivot(index='layer', columns='head', values=AttentionsConstants.CORRELATION)
        ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1, ax=ax,
                         annot_kws={"fontsize": 11, "fontweight": 'bold'})
        title = f'Correlation between the attention matrices by comparing head to head for the text:\n ' \
                f'{sample.text}\n' \
                f'Using the models: 1.{model_name1}. 2.{model_name2}'
        # add the title (wrap it doesn't get cut off)
        ax.set_title(title, fontsize=14, fontweight='bold', wrap=True)

        ax.set_xlabel('Head')
        ax.set_ylabel('Layer')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        # Show the plot
        plt.show()

    @staticmethod
    def plot_histogram_of_layers_and_heads(correlations_attentions_comparisons: CorrelationsAttentionsComparisons):
        """
        Gets data of a sample of a shape (L,H,L,H)
        shows histogram of the correlation of layer to layer compare to other heads.
        print the mean of those groups.
        """
        full_correlations_attentions_comparisons = correlations_attentions_comparisons.get_full_correlations_comparisons()
        group1 = full_correlations_attentions_comparisons[np.arange(full_correlations_attentions_comparisons.shape[0]),
                 :, np.arange(full_correlations_attentions_comparisons.shape[0]), :].flatten()
        mask = np.ones_like(full_correlations_attentions_comparisons)
        mask[np.arange(full_correlations_attentions_comparisons.shape[0]), :,
        np.arange(full_correlations_attentions_comparisons.shape[0]), :] = 0
        group2 = full_correlations_attentions_comparisons[mask.astype(bool)]

        min_data = full_correlations_attentions_comparisons.min()
        max_data = full_correlations_attentions_comparisons.max()
        bins = np.linspace(min_data, max_data, 50)

        plt.hist(group1, bins, alpha=0.5, label='Layer to Layer corr')
        plt.hist(group2, bins, alpha=0.5, label='Other heads')
        plt.legend(loc='upper right')
        plt.show()

        print(f"Layer to Layer mean corr:{group1.mean():.3f}")
        print(f"Other heads mean corr:{group2.mean():.3f}")
