import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
import seaborn as sns

from AttentionsComparators.AttentionsComparator import AttentionsComparator
from AttentionExtractors import ExtractorFactory
from CorrelationAnalysis import CorrelationAnalysis
from Common.Constants import Constants

DEFULT_AUDIO_KEY = Constants.AudioModelProcessorConstants.LIBRISPEECH_AUDIO_KEY


class StatsGenerator:
    def __init__(self, model1_dict: dict, model2_dict: dict, metric: str = 'Cosine'):
        self.metric = metric
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor1 = ExtractorFactory(**model1_dict, device=self.device)
        self.extractor2 = ExtractorFactory(**model2_dict, device=self.device)
        self.comparator = AttentionsComparator(correlation_analysis=CorrelationAnalysis(metric=metric))

    def get_attentions(self, sample):
        attention_model1 = self.extractor1.extract_attention(sample=sample, audio_key=DEFULT_AUDIO_KEY)
        attention_model2 = self.extractor2.extract_attention(sample=sample, audio_key=DEFULT_AUDIO_KEY)

        if self.extractor1.type != self.extractor2.type:
            attention_model1 = self.extractor1.align_attentions(sample, attention_model1)
            attention_model2 = self.extractor2.align_attentions(sample, attention_model2)

        return attention_model1, attention_model2

    def avg_by_layer_sample(self, sample: dict):
        attention_model1, attention_model2 = self.get_attentions(sample)

        # Mean all heads in the same layer. Shape [L, 1, tokens, tokens]
        avg_by_layer_model1 = np.mean(attention_model1.attentions, axis=1)[:, None]
        avg_by_layer_model2 = np.mean(attention_model2.attentions, axis=1)[:, None]

        data = self.comparator.compare_matrices(avg_by_layer_model1, avg_by_layer_model2).squeeze()

        #TODO: Move it from here, that's here just for now.
        df_data = pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=(10, 12))
        ax = sns.heatmap(df_data, annot=True, cmap='coolwarm', vmin=0, vmax=1, ax=ax,
                         annot_kws={"fontsize": 11, "fontweight": 'bold'})
        title = f'Correlation between the attention matrices for the text:\n ' \
                f'{sample["text"]}\n' \
                f'Using the models: 1.{self.extractor1.model_name}. 2.{self.extractor2.model_name}'
        # add the title (wrap it doesn't get cut off)
        ax.set_title(title, fontsize=14, fontweight='bold', wrap=True)

        ax.set_xlabel('Layers Model 1')
        ax.set_ylabel('Layers Model 2')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        # Save the plot
        # ax.figure.savefig(f'../Results/Correlation/{model_name1}_{model_name2}_{text}.png')
        # Show the plot
        plt.show()

        return data

    def get_all_data_from_sample(self, sample: dict):
        attention_model1, attention_model2 = self.get_attentions(sample)
        data = self.comparator.compare_matrices(attention_model1, attention_model2)
        return data

    def get_all_data_head_to_head_sample(self, sample: dict):
        attention_model1, attention_model2 = self.get_attentions(sample)
        data = self.comparator.compare_head_to_head(attention_model1, attention_model2)
        self.comparator.plot_correlation_result_hitmap(sample['text'], self.extractor1.model_name,
                                                       self.extractor2.model_name, data)
        return data



if __name__ == '__main__':
    dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", 'clean', split='validation')
    sample = dataset[0]
    # sample['text'] = 'Hello, my dog is cute'
    model1_dict = dict(model_name="bert-base-uncased", type='text')
    model2_dict = dict(model_name="facebook/wav2vec2-base-960h", type='audio')
    model3_dict = dict(model_name="roberta-base", type='text')

    stats_generator = StatsGenerator(model1_dict, model2_dict)
    data = stats_generator.avg_by_layer_sample(sample)
    data_heads = stats_generator.get_all_data_head_to_head_sample(sample)

    stats_generator = StatsGenerator(model1_dict, model3_dict)
    data1 = stats_generator.avg_by_layer_sample(sample)
    data_heads1 = stats_generator.get_all_data_head_to_head_sample(sample)


    # for i in range(10):
    #     print(i)
    #     data += reporter.avg_by_layer_sample(dataset[1])
    #
    # data = data / 10
