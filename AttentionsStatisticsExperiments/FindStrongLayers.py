import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm

from Common.Constants import Constants
from DataModels.CorrelationsAttentionsComparisons import CorrelationsAttentionsComparisons

DEFAULT_AUDIO_KEY = Constants.AudioModelProcessorConstants.LIBRISPEECH_AUDIO_KEY


class FindStrongLayers:
    """
    This class is used to create pipeline of comparisons with standard deviation between attentions of two models.
    """

    def __init__(self, threshold: float, top_k: int):
        self.threshold = threshold
        self.top_k = top_k

    def count_entries_above_threshold(self,
                                      correlations_attentions_comparisons: CorrelationsAttentionsComparisons) -> np.ndarray:
        num_of_heads = correlations_attentions_comparisons.heads_model_1
        num_of_layers = correlations_attentions_comparisons.layers_model_1
        count_entries_above_threshold_model = np.zeros(
            (num_of_layers, num_of_heads, num_of_layers, num_of_heads)).astype(int)
        for layer in range(num_of_layers):
            for head in range(num_of_heads):
                correlation_of_head = correlations_attentions_comparisons.get(layer, head, None, None)
                # remove the 2 first dimensions of the correlation matrix
                correlation_of_head = correlation_of_head.squeeze(0).squeeze(0)
                layers_above_threshold, heads_above_threshold = np.where(correlation_of_head > self.threshold)
                # for each pair of max_layers, max_heads increment the count_max_layers_and_heads in the same position
                for max_layer, max_head in zip(layers_above_threshold, heads_above_threshold):
                    count_entries_above_threshold_model[layer, head, max_layer, max_head] += 1
        return count_entries_above_threshold_model

    def get_sum_all_examples_entries_above_threshold_model(self, correlations):
        # take the values of the correlations of the all the samples and concatenate them to one matrix
        all_examples_entries_above_threshold_model = {}
        for i, (id, corr) in tqdm(enumerate(correlations.items())):
            count_entries_above_threshold_model = self.count_entries_above_threshold(corr)
            all_examples_entries_above_threshold_model[id] = count_entries_above_threshold_model
        return all_examples_entries_above_threshold_model

    def run(self, saved_corr_path: str):
        correlations = self.load_saved_correlations(saved_corr_path)
        all_examples_entries_above_threshold_model1 = self.get_sum_all_examples_entries_above_threshold_model(
            correlations)

        # save to pickle
        sum_all_examples_entries_above_threshold_model1, strong_layers_model1 = self.find_strong_layers(
            all_examples_entries_above_threshold_model1, top_k=self.top_k)
        num_of_examples = len(correlations)
        return sum_all_examples_entries_above_threshold_model1, strong_layers_model1, num_of_examples

    def find_strong_layers(self, all_examples_entries_above_threshold_model, top_k):
        # take the values of the correlations of the all the samples and concatenate them to one matrix
        sum_all_examples_entries_above_threshold_model = sum(list(all_examples_entries_above_threshold_model.values()))
        # find the 3 top k values in each matrix
        top_k_model = self.find_top_k(sum_all_examples_entries_above_threshold_model, k=top_k)
        return sum_all_examples_entries_above_threshold_model, top_k_model

    def find_top_k(self, sum_all_examples_entries_above_threshold_model, k):
        # find the 3 top k values in each matrix
        top_k_results = np.unravel_index(
            np.argpartition(sum_all_examples_entries_above_threshold_model, -k, axis=None)[-k:],
            sum_all_examples_entries_above_threshold_model.shape)

        return top_k_results

    def print_results(self, sum_all_examples_entries_above_threshold_model1, strong_layers_model1, num_exampls,
                      model_name: str):
        print(f"Strong layers for model{model_name}:")
        for l1, h1, l2, h2, in zip(strong_layers_model1[0], strong_layers_model1[1], strong_layers_model1[2],
                                   strong_layers_model1[3]):
            # print(f"Layer1: {l1}, Head1: {h1}, Layer2: {l2}, Head2: {h2}")
            print(f"Layers: {l1} and {l2}, Heads: {h1} and {h2}")
            num_examples_above_th = sum_all_examples_entries_above_threshold_model1[l1, h1, l2, h2] / num_exampls
            # trun to 2 digits after the decimal point
            print(f'num of examples above threshold: {num_examples_above_th:.2f}')

    def load_saved_correlations(self, saved_corr_path: str):
        # load pickle file
        with open(saved_corr_path, 'rb') as handle:
            correlations = pickle.load(handle)
        return correlations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_corr_path", type=str, choices=[
        'correlations_for_bert-base-uncased_and_facebookwav2vec2-base-960h.pickle',
        'correlations_for_bert-base-uncased_and_roberta-base.pickle'],
                        default=r'correlations_for_bert-base-uncased_and_facebookwav2vec2-base-960h.pickle')
    parser.add_argument("--top_k", type=int, default=5, help="The number of top layers and heads to find")
    parser.add_argument("--threshold", type=float, default=0.9, help="The threshold for the correlation")

    args = parser.parse_args()

    find_strong_layers = FindStrongLayers(threshold=args.threshold, top_k=args.top_k)
    sum_all_examples_entries_above_threshold_model1, strong_layers_model1, num_exampls = find_strong_layers.run(
        args.saved_corr_path)
    print(f"Experiment: {args.saved_corr_path}")
    find_strong_layers.print_results(sum_all_examples_entries_above_threshold_model1, strong_layers_model1,
                                     num_exampls, 1)
    print("==========================================")
