import argparse
import pickle
from typing import Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from AttentionsAnalysis.AnalysisGenerator import AnalysisGenerator
from DataModels.CorrelationsAttentionsComparisons import CorrelationsAttentionsComparisons
from DataModels.DataType import DataType
from DataModels.ModelMetadata import ModelMetadata
from DataModels.Sample import Sample
import matplotlib.pyplot as plt

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
    def print_important_tages(self, all_data_th, ind_by_rel=None):
        mean_above_threshold = np.mean(all_data_th, axis=0)
        # Here I print the mean of the number of Samples above the threshold.
        important_tages = {0: {'name': 'advmod', 'loc': (0,10,4,1)},
         1: {'name': 'aux', 'loc': (6,9,7,10)},
         2: {'name': 'case', 'loc': (7,10,7,5)},
         3: {'name': 'det', 'loc': (1,1,8,1)},
         4: {'name': 'fixed', 'loc': (3,5,3,4)},
         5: {'name': 'cop', 'loc': (6,5,7,5)},
         6: {'name': 'obj', 'loc': (7,9,7,11)},
         7: {'name': 'conj', 'loc': (4,3,7,4)}}

        for index, tag_dat in important_tages.items():
            shape = tag_dat['loc']
            name = tag_dat['name']
            print(f"{name}")
            print(f"ALL {name} {shape} mean above the threshold: {mean_above_threshold[shape].round(3)}")
            if ind_by_rel is not None:
                heads_with_tag = np.sum(all_data_th[:, shape[0], shape[1], shape[2], shape[3]] & ind_by_rel[:,index])
                heads_without_tag = np.sum(all_data_th[:, shape[0], shape[1], shape[2], shape[3]] & (-1* (ind_by_rel[:,index] - 1)))
                print(f"Only WITH tag {name} {shape} mean above the threshold: {((heads_with_tag)/np.sum(ind_by_rel[:,index])).round(3)}")
                print(f"Only WITHOUT tag {name} {shape} mean above the threshold: {(heads_without_tag/np.sum((-1* (ind_by_rel[:,index] - 1)))).round(3)}")

        t=1
    def explore_high_correlations(self, mean_above_threshold):
        # Here we print some metrcies with the highest correlation:
        ls1, hs1, ls2, hs2 = self.largest_indices(mean_above_threshold, 1500)
        flhs1, flhs2 = [], []
        fls1, fhs1, fls2, fhs2 = [], [], [], []

        dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", 'clean', split='validation', )
        model1_metadata = ModelMetadata(model_name="bert-base-uncased", data_type=DataType.Text,
                                        align_tokens_to_bert_tokens=False, use_cls_and_sep=True)
        model2_metadata = ModelMetadata(model_name="facebook/wav2vec2-base-960h", data_type=DataType.Audio,
                                        align_tokens_to_bert_tokens=True, use_cls_and_sep=True)
        analysis_generator = AnalysisGenerator(model1_metadata, model2_metadata, metric='Cosine')
        attentions_model1, attentions_model2 = [], []
        for i in tqdm(range(5)):
            sample1 = Sample(id=dataset[i]["id"], text=dataset[i]["text"], audio=dataset[i]["audio"])
            sample2 = sample1
            attention_model1, attention_model2 = analysis_generator.get_attentions(sample1, sample2)
            attentions_model1.append(attention_model1)
            attentions_model2.append(attention_model2)
        bad_list1, bad_list2 = [], []
        for i in range(len(ls1)):
            print(i)
            if ((ls1[i], hs1[i]) in bad_list1) and ((ls2[i], hs2[i]) in bad_list2):
                # bad_list1.append((ls1[i], hs1[i]))
                # bad_list2.append((ls2[i], hs2[i]))
                continue

            fig, axs = plt.subplots(2, 5)
            ims = []
            for j in range(5):
                ims.append(axs[0, j].imshow(attentions_model1[j][ls1[i], hs1[i]]))
                ims.append(axs[1, j].imshow(attentions_model2[j][ls2[i], hs2[i]]))

            plt.show()
            val = input("Is those a intersting N\Y? : ")
            if val.lower() == 'n':
                bad_list1.append((ls1[i], hs1[i]))
                bad_list2.append((ls2[i], hs2[i]))
                continue
            flhs1.append((ls1[i], hs1[i]))
            flhs2.append((ls2[i], hs2[i]))
            fls1.append(ls1[i])
            fhs1.append(hs1[i])
            fls2.append(ls2[i])
            fhs2.append(hs2[i])

        ls1 = np.array(fls1)
        hs1 = np.array(fhs1)
        ls2 = np.array(fls2)
        hs2 = np.array(fhs2)
        best_mean_corr = mean_above_threshold[(ls1, hs1, ls2, hs2)]
        for i in range(len(ls1)):
            print(f"Check those matrices index: [{ls1[i], hs1[i], ls2[i], hs2[i]}], mean corr: {best_mean_corr[i].round(4)}")

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
        # correlations_for_model1 = {}
        # correlations_for_model2 = {}
        #
        # for id, corr in tqdm(correlations.items()):
        #     maximum_correlations_for_model1, maximum_correlations_for_model2 = self.find_maximum_correlations(corr)
        #     correlations_for_model1[id] = maximum_correlations_for_model1
        #     correlations_for_model2[id] = maximum_correlations_for_model2
        #
        # # find the mean of the correlations for each model and standard deviation
        # mean_correlations_for_model1, standard_deviation_for_model1 = self.calculate_mean_correlations(
        #     correlations_for_model1)
        # mean_correlations_for_model2, standard_deviation_for_model2 = self.calculate_mean_correlations(
        #     correlations_for_model2)
        # print(
        #     f"mean correlations for model1 on {len(correlations_for_model1)} examples: {mean_correlations_for_model1} +- {standard_deviation_for_model1}")
        # print(
        #     f"mean correlations for model2 on {len(correlations_for_model2)} examples: {mean_correlations_for_model2} +- {standard_deviation_for_model2}")

        all_data = np.zeros((len(correlations), 12, 12, 12, 12))
        for i, (id, corr) in enumerate(tqdm(correlations.items())):
            all_data[i] = corr.full_correlations_comparisons
        mean_all_corrs = all_data.mean().round(3)
        std_all_corrs = all_data.std().round(3)

        print(f"mean ALL correlations for an {len(correlations)} examples: {mean_all_corrs} +- {std_all_corrs}")
        th = mean_all_corrs + std_all_corrs
        all_data_th = np.ma.masked_where(all_data > th, all_data).mask
        mean_above_threshold = np.mean(all_data_th, axis=0)
        print(f"Use threshold for determine corr: {th}")

        ind_by_rel = np.load("indices_by_rel.npy")
        ind_by_rel = ind_by_rel if ind_by_rel.shape[0] == all_data_th.shape[0] else None
        self.print_important_tages(all_data_th, ind_by_rel)
        # self.explore_high_correlations(mean_above_threshold)

    def load_saved_correlations(self):
        # load pickle file
        with open(self.saved_correlations_path, 'rb') as handle:
            correlations = pickle.load(handle)
        return correlations

    def get_saved_correlations_path(self):
        if self.experiment_name == "text_to_text":
            correlations_file = 'correlations_for_bert-base-uncased_and_roberta-base.pickle'
        elif self.experiment_name == "text_to_audio":
            correlations_file = 'correlations_for_bert-base-uncased_and_facebookwav2vec2-base-960h_0_None_jaccard.pkl'
        elif self.experiment_name == "audio_to_audio":
            raise Exception("Experiment audio_to_audio currently not supported")
        else:
            raise Exception("Experiment name is not valid")

        return correlations_file

    @staticmethod
    def largest_indices(ary, n):
        """Returns the n largest indices from a numpy array."""
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, ary.shape)


if __name__ == "__main__":
    # add arguments from command line with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, choices=["text_to_text", "audio_to_audio", "text_to_audio"],
                        default="text_to_audio", help="The name of the experiment to run")
    args = parser.parse_args()

    attention_similarity = AttentionsStatisticsExperiment(args.experiment_name)
    attention_similarity.run()
