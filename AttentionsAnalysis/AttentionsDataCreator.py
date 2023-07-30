import argparse
import pickle

import torch
from datasets import load_dataset
from tqdm import tqdm

from AnalysisGenerator import AnalysisGenerator
from DataModels.DataType import DataType
from DataModels.ModelMetadata import ModelMetadata
from DataModels.Sample import Sample
import os
import pandas as pd


bert_base_path = "/home/vpnuser/cs_huji/anlp/ud_bert_attn"
w2v2_base_path = "/home/vpnuser/cs_huji/anlp/ud_w2v2_attn_agg_tokens"



class AttentionsDataCreator:
    """
    This class is used to create pipeline of comparisons with standard deviation between attentions of two models.
    """

    def __init__(self, model1_metadata: ModelMetadata, model2_metadata: ModelMetadata, use_dummy_dataset: bool = False,
                 metric: str = "Cosine"):
        self.model1_metadata = model1_metadata
        self.model2_metadata = model2_metadata
        self.metric = metric
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.analysis_generator = AnalysisGenerator(model1_metadata, model2_metadata, metric=metric)
        self.dataset = self.load_dummy_dataset() if use_dummy_dataset else self.load_dataset()

    def get_correlations_attentions_comparisons(self, attention_model1, attention_model2):
        # attention_model1, attention_model2 = self.analysis_generator.get_attentions(sample1, sample2)
        correlations_attentions_comparisons = self.analysis_generator.get_correlations_of_attentions(attention_model1,
                                                                                                     attention_model2)
        return correlations_attentions_comparisons

    def run(self):
        correlations = {}
        for i, fname in tqdm(enumerate(self.dataset)):
            attn1 = pd.read_pickle(os.path.join(bert_base_path, fname))
            attn2 = pd.read_pickle(os.path.join(w2v2_base_path, fname))
            try:
                correlations_attentions_comparisons = self.get_correlations_attentions_comparisons(
                    attn1, attn2)
                correlations[fname.split('.')[0]] = correlations_attentions_comparisons
            except AssertionError as e:
                example = f"{i} {fname}"
                print(f"Failed to calculate for sample {example} with error {e}")
        # save results to pickle file
        with open(f'correlations_for_{self.model1_metadata.model_name}_and_{self.model2_metadata.model_name.replace("/", "_")}_transpose.pickle',
                  'wb') as handle:
            pickle.dump(correlations, handle)

    def load_dataset(self):
        # dataset = load_dataset("librispeech_asr", 'clean', split='validation')
        # return dataset
        bert_files = os.listdir(bert_base_path)
        bert_files.sort(key=lambda x: int(x.split("_")[0]))

        return bert_files

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
    parser.add_argument("--metric", type=str, default='Cosine',
                        help="Which metric to use")

    args = parser.parse_args()

    if args.experiment_name == "text_to_text":
        model1_metadata = ModelMetadata(model_name="bert-base-uncased", data_type=DataType.Text,
                                        align_tokens_to_bert_tokens=False, use_cls_and_sep=True)
        model2_metadata = ModelMetadata(model_name="roberta-base", data_type=DataType.Text,
                                        align_tokens_to_bert_tokens=False, use_cls_and_sep=True)
        attention_similarity = AttentionsDataCreator(model1_metadata, model2_metadata,
                                                     use_dummy_dataset=args.use_dummy_dataset,
                                                     metric=args.metric)
        attention_similarity.run()

    elif args.experiment_name == "text_to_audio":
        model1_metadata = ModelMetadata(model_name="bert-base-uncased", data_type=DataType.Text,
                                        align_tokens_to_bert_tokens=False, use_cls_and_sep=True)
        model2_metadata = ModelMetadata(model_name="facebook/wav2vec2-base-960h", data_type=DataType.Audio,
                                        align_tokens_to_bert_tokens=True, use_cls_and_sep=True)
        attention_similarity = AttentionsDataCreator(model1_metadata, model2_metadata,
                                                     use_dummy_dataset=args.use_dummy_dataset,
                                                     metric=args.metric)
        attention_similarity.run()


    elif args.experiment_name == "audio_to_audio":
        print("currently not supported")

    else:
        raise Exception("Experiment name not supported")
