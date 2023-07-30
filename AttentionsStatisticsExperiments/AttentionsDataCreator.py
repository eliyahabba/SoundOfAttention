import argparse
import pickle

import torch
from datasets import load_dataset
from tqdm import tqdm
import sys
sys.path.append(r'/cs/snapless/gabis/eliyahabba/Legal_nlp/SoundOfAttention')

from AttentionsAnalysis.AnalysisGenerator import AnalysisGenerator
from DataModels.DataType import DataType
from DataModels.ModelMetadata import ModelMetadata
from DataModels.Sample import Sample


class AttentionsDataCreator:
    """
    This class is used to create pipeline of comparisons with standard deviation between attentions of two models.
    """

    def __init__(self, model1_metadata: ModelMetadata, model2_metadata: ModelMetadata, use_dummy_dataset: bool = False,
                 start_example=None, end_example=None,
                 metric: str = "Cosine"):
        self.model1_metadata = model1_metadata
        self.model2_metadata = model2_metadata
        self.metric = metric
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.analysis_generator = AnalysisGenerator(model1_metadata, model2_metadata, metric='Cosine')
        self.start_example = start_example
        self.end_example = end_example
        self.dataset = self.load_dummy_dataset() if use_dummy_dataset else self.load_dataset(self.start_example,
                                                                                             self.end_example)

    def get_correlations_attentions_comparisons(self, sample1: Sample, sample2: Sample):
        attention_model1, attention_model2 = self.analysis_generator.get_attentions(sample1, sample2)
        return attention_model1, attention_model2

    def run(self):
        all_attention_model1 = {}
        all_attention_model2 = {}
        for i in tqdm(range(len(self.dataset))):
            sample1 = Sample(id=self.dataset[i]["id"], text=self.dataset[i]["text"], audio=self.dataset[i]["audio"])
            sample2 = sample1
            try:
                attention_model1, attention_model2 = self.get_correlations_attentions_comparisons(sample1, sample2)
                all_attention_model1[sample1.id] = attention_model1
                all_attention_model2[sample2.id] = attention_model2
            except AssertionError as e:
                example = f"{sample1.text}" if sample1.text == sample2.text else f"{sample1.text} and {sample2.text}"
                print(f"Failed to calculate for sample {example}")
        # save results to pickle file
        print(f"the number of samples in model1 is {len(all_attention_model1)} and the number of samples is {len(all_attention_model2)}")
        self.save_correlations(all_attention_model1, all_attention_model2)

    def load_dataset(self, start_example=None, end_example=None):
        if start_example is not None and end_example is not None:
            dataset = load_dataset("librispeech_asr", 'clean', split=f'validation[{start_example}:{end_example}]')
        else:
            dataset = load_dataset("librispeech_asr", 'clean', split='validation')
        return dataset

    def load_dummy_dataset(self):
        dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", 'clean', split='validation')
        return dataset

    def save_correlations(self, attention_model1, attention_model2):
        model_name1 = self.model1_metadata.model_name
        model_name2 = self.model2_metadata.model_name
        # if the model name is 'facebook/wav2vec2-base-960h' we need to remove the '/' from the name
        if '/' in model_name1:
            model_name1 = model_name1.replace('/', '')
        if '/' in model_name2:
            model_name2 = model_name2.replace('/', '')
        path = f'attention_for_{model_name1}{self.start_example}_{self.end_example}.pkl'
        with open(path, 'wb') as handle:
            pickle.dump(attention_model1, handle, protocol=pickle.HIGHEST_PROTOCOL)

        path = f'attention_for_{model_name2}{self.start_example}_{self.end_example}.pkl'
        with open(path, 'wb') as handle:
            pickle.dump(attention_model2, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    # add arguments from command line with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, choices=["text_to_text", "audio_to_audio", "text_to_audio"],
                        default="text_to_audio", help="The name of the experiment to run")
    parser.add_argument("--use_dummy_dataset", type=bool, default=False,
                        help="Whether to use a dummy dataset for the experiment")
    parser.add_argument("--start_example", type=int, default=0)
    parser.add_argument("--end_example", type=int, default=400)
    args = parser.parse_args()

    if args.experiment_name == "text_to_text":
        model1_metadata = ModelMetadata(model_name="bert-base-uncased", data_type=DataType.Text,
                                        align_tokens_to_bert_tokens=False, use_cls_and_sep=True)
        model2_metadata = ModelMetadata(model_name="roberta-base", data_type=DataType.Text,
                                        align_tokens_to_bert_tokens=False, use_cls_and_sep=True)

    elif args.experiment_name == "text_to_audio":
        model1_metadata = ModelMetadata(model_name="bert-base-uncased", data_type=DataType.Text,
                                        align_tokens_to_bert_tokens=False, use_cls_and_sep=True)
        model2_metadata = ModelMetadata(model_name="facebook/wav2vec2-base-960h", data_type=DataType.Audio,
                                        align_tokens_to_bert_tokens=True, use_cls_and_sep=True)

    elif args.experiment_name == "audio_to_audio":
        raise Exception("Experiment audio_to_audio currently not supported")

    else:
        raise Exception("Experiment name is not valid")

    attention_similarity = AttentionsDataCreator(model1_metadata, model2_metadata,
                                                 use_dummy_dataset=args.use_dummy_dataset,
                                                 start_example=args.start_example, end_example=args.end_example)

    attention_similarity.run()
