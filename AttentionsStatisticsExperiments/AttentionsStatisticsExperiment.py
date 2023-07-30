import argparse
import pickle
from typing import Tuple

import numpy as np
from tqdm import tqdm

from DataModels.CorrelationsAttentionsComparisons import CorrelationsAttentionsComparisons


class AttentionsStatisticsExperiment:
    """
    This class is used to create pipeline of comparisons with standard deviation between attentions of two models.
    """

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.saved_correlations_path = self.get_saved_correlations_path()

    def find_maximum_correlations(self, correlations_attentions_comparisons: CorrelationsAttentionsComparisons):
        H = correlations_attentions_comparisons.heads_model_1
        L = correlations_attentions_comparisons.layers_model_1

        # for each head in for each layer find the head with the highest correlation in the same layer in the other model
        # and the head with the lowest correlation in the same layer in the other model
        maximum_correlations_for_model1 = np.zeros((L, H))
        for l in range(L):
            for h in range(H):
                correlation_of_head = correlations_attentions_comparisons.get(l, h, None, None)
                # find the head with the highest correlation in the same layer in the other model
                max_correlation = correlation_of_head.max()
                maximum_correlations_for_model1[l, h] = max_correlation

        maximum_correlations_for_model2 = np.zeros((L, H))
        for l in range(L):
            for h in range(H):
                correlation_of_head = correlations_attentions_comparisons.get(None, None, l, h)
                # find the head with the highest correlation in the same layer in the other model
                max_correlation = correlation_of_head.max()
                maximum_correlations_for_model2[l, h] = max_correlation
        return maximum_correlations_for_model1, maximum_correlations_for_model2

    def calculate_mean_correlations(self, correlations_for_model: dict) -> Tuple[float, float]:
        # find the mean of the correlations for each model and standard deviation
        # first flatten the correlations_for_model1 from all the samples
        correlations_for_model1_flattened = np.array(list(correlations_for_model.values())).flatten()
        mean_correlations_for_model1 = np.mean(correlations_for_model1_flattened)
        std_correlations_for_model1 = np.std(correlations_for_model1_flattened)
        # truncate the mean and std to 2 decimal places
        mean_correlations_for_model1 = np.trunc(mean_correlations_for_model1 * 100) / 100
        std_correlations_for_model1 = np.trunc(std_correlations_for_model1 * 100) / 100
        return mean_correlations_for_model1, std_correlations_for_model1

    def run(self):
        correlations = self.load_saved_correlations()
        correlations_for_model1 = {}
        correlations_for_model2 = {}
        for id, corr in tqdm(correlations.items()):
            maximum_correlations_for_model1, maximum_correlations_for_model2 = self.find_maximum_correlations(corr)
            correlations_for_model1[id] = maximum_correlations_for_model1
            correlations_for_model2[id] = maximum_correlations_for_model2

        # find the mean of the correlations for each model and standard deviation
        mean_correlations_for_model1, standard_deviation_for_model1 = self.calculate_mean_correlations(
            correlations_for_model1)
        mean_correlations_for_model2, standard_deviation_for_model2 = self.calculate_mean_correlations(
            correlations_for_model2)
        print(
            f"mean correlations for model1 on {len(correlations_for_model1)} examples: {mean_correlations_for_model1} +- {standard_deviation_for_model1}")
        print(
            f"mean correlations for model2 on {len(correlations_for_model2)} examples: {mean_correlations_for_model2} +- {standard_deviation_for_model2}")

    def load_saved_correlations(self):
        # load pickle file
        with open(self.saved_correlations_path, 'rb') as handle:
            correlations = pickle.load(handle)
        return correlations

    def get_saved_correlations_path(self):
        if self.experiment_name == "text_to_text":
            correlations_file = 'correlations_for_bert-base-uncased_and_roberta-base.pickle'
        elif self.experiment_name == "text_to_audio":
            correlations_file = 'correlations_for_bert-base-uncased_and_facebookwav2vec2-base-960h.pickle'
        elif self.experiment_name == "audio_to_audio":
            raise Exception("Experiment audio_to_audio currently not supported")
        else:
            raise Exception("Experiment name is not valid")
        return correlations_file


if __name__ == "__main__":
    # add arguments from command line with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, choices=["text_to_text", "audio_to_audio", "text_to_audio"],
                        default="text_to_audio", help="The name of the experiment to run")
    args = parser.parse_args()

    attention_similarity = AttentionsStatisticsExperiment(args.experiment_name)
    attention_similarity.run()
