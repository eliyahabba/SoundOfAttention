import argparse
import pickle
from typing import Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from AttentionsAnalysis.AnalysisGenerator import AnalysisGenerator
from AttentionsComparators.CorrelationsAttentionsComparator import AttentionsConstants
from Common.Constants import Constants
from DataModels.DataType import DataType
from DataModels.ModelMetadata import ModelMetadata
from DataModels.Sample import Sample

DEFAULT_AUDIO_KEY = Constants.AudioModelProcessorConstants.LIBRISPEECH_AUDIO_KEY


class AttentionsStatisticsExperiment:
    """
    This class is used to create pipeline of comparisons with standard deviation between attentions of two models.
    """

    def __init__(self, model1_metadata: ModelMetadata, model2_metadata: ModelMetadata, use_dummy_dataset: bool = False,
                 metric: str = "Cosine"):
        self.model1_metadata = model1_metadata
        self.model2_metadata = model2_metadata
        self.metric = metric
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.analysis_generator = AnalysisGenerator(model1_metadata, model2_metadata, metric='Cosine')
        self.dataset = self.load_dummy_dataset() if use_dummy_dataset else self.load_dataset()

    def find_maximum_correlations(self, sample1: Sample, sample2: Sample):
        attention_model1, attention_model2 = self.analysis_generator.get_attentions(sample1, sample2)
        correlations_attentions_comparisons = self.analysis_generator.get_correlations_of_attentions(attention_model1,
                                                                                                     attention_model2)
        L = attention_model1.shape[AttentionsConstants.LAYER_AXIS]
        H = attention_model2.shape[AttentionsConstants.HEAD_AXIS]

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
        correlations_for_model1 = {}
        correlations_for_model2 = {}
        for i in tqdm(range(len(self.dataset))):
            sample1 = Sample(id=self.dataset[i]["id"], text=self.dataset[i]["text"], audio=self.dataset[i]["audio"])
            sample2 = sample1
            try:
                maximum_correlations_for_model1, maximum_correlations_for_model2 = self.find_maximum_correlations(
                    sample1, sample2)
                correlations_for_model1[sample1.id] = maximum_correlations_for_model1
                correlations_for_model2[sample2.id] = maximum_correlations_for_model2
            except AssertionError as e:
                example = f"{sample1.text}" if sample1.text == sample2.text else f"{sample1.text} and {sample2.text}"
                print(f"Failed to calculate for sample {example}")
        # save results to pickle file
        with open(f'correlations_for_{self.model1_metadata.model_name}_with{self.model2_metadata.model_name}.pickle',
                  'wb') as handle:
            pickle.dump(correlations_for_model1, handle)
        with open(f'correlations_for_{self.model2_metadata.model_name}_with{self.model1_metadata.model_name}.pickle',
                  'wb') as handle:
            pickle.dump(correlations_for_model2, handle)

        # find the mean of the correlations for each model and standard deviation
        mean_correlations_for_model1, standard_deviation_for_model1 = self.calculate_mean_correlations(
            correlations_for_model1)
        mean_correlations_for_model2, standard_deviation_for_model2 = self.calculate_mean_correlations(
            correlations_for_model2)
        print(
            f"mean correlations for model1 on {len(correlations_for_model1)} examples: {mean_correlations_for_model1} +- {standard_deviation_for_model1}")
        print(
            f"mean correlations for model2 on {len(correlations_for_model2)} examples: {mean_correlations_for_model2} +- {standard_deviation_for_model2}")

    def load_dataset(self):
        dataset = load_dataset("librispeech_asr", 'clean', split='validation')
        return dataset

    def load_dummy_dataset(self):
        dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", 'clean', split='validation')
        return dataset


if __name__ == "__main__":
    # add arguments from command line with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, choices=["text_to_text", "audio_to_audio", "text_to_audio"],
                        default="text_to_audio", help="The name of the experiment to run")
    parser.add_argument("--use_dummy_dataset", type=bool, default=False,
                        help="Whether to use a dummy dataset for the experiment")
    args = parser.parse_args()

    if args.experiment_name == "text_to_text":
        model1_metadata = ModelMetadata(model_name="bert-base-uncased", data_type=DataType.Text,
                                        align_tokens_to_bert_tokens=False, use_cls_and_sep=True)
        model2_metadata = ModelMetadata(model_name="roberta-base", data_type=DataType.Text,
                                        align_tokens_to_bert_tokens=False, use_cls_and_sep=True)
        attention_similarity = AttentionsStatisticsExperiment(model1_metadata, model2_metadata,
                                                              use_dummy_dataset=args.use_dummy_dataset)
        attention_similarity.run()

    elif args.experiment_name == "text_to_audio":
        model1_metadata = ModelMetadata(model_name="bert-base-uncased", data_type=DataType.Text,
                                        align_tokens_to_bert_tokens=False, use_cls_and_sep=True)
        model2_metadata = ModelMetadata(model_name="facebook/wav2vec2-base-960h", data_type=DataType.Audio,
                                        align_tokens_to_bert_tokens=True, use_cls_and_sep=True)
        attention_similarity = AttentionsStatisticsExperiment(model1_metadata, model2_metadata,
                                                              use_dummy_dataset=args.use_dummy_dataset)
        attention_similarity.run()


    elif args.experiment_name == "audio_to_audio":
        print("currently not supported")
