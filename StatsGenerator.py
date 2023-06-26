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

    def get_attentions(self, sample, sample2=None):
        attention_model1 = self.extractor1.extract_attention(sample=sample, audio_key=DEFULT_AUDIO_KEY)
        if sample2:
            attention_model2 = self.extractor2.extract_attention(sample=sample2, audio_key=DEFULT_AUDIO_KEY)
        else:
            attention_model2 = self.extractor2.extract_attention(sample=sample, audio_key=DEFULT_AUDIO_KEY)

        if self.extractor1.type != self.extractor2.type:
            attention_model1 = self.extractor1.align_attentions(sample, attention_model1)
            attention_model2 = self.extractor2.align_attentions(sample, attention_model2)

        assert attention_model1.shape == attention_model2.shape

        return attention_model1, attention_model2

    def avg_by_layer_sample(self, sample: dict, sample2: dict = None,
                            attention_model1 = None, attention_model2 = None, display: bool = True):
        """
        Does average on all head of each layer and produce a Layers X Layers matrix of avg
        correlation between the models.

        sample - to run the models on.
        sample2 - if provided the second model runs on this sample.
        attention_model1, attention_model2 - if provided, the func won't run the models.
        display - whether to display or just return the data.
        """
        if not attention_model1 or not attention_model2:
            attention_model1, attention_model2 = self.get_attentions(sample, sample2)

        # Mean all heads in the same layer. Shape [L, 1, tokens, tokens]
        avg_by_layer_model1 = np.mean(attention_model1.attentions, axis=1)[:, None]
        avg_by_layer_model2 = np.mean(attention_model2.attentions, axis=1)[:, None]

        data = self.comparator.compare_matrices(avg_by_layer_model1, avg_by_layer_model2).squeeze()

        if display:
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
            # Show the plot
            plt.show()

        return data

    def get_all_data_from_sample(self, sample: dict, sample2: dict = None,
                                 attention_model1 = None, attention_model2 = None, display: bool = True):
        """
        gets the whole data is possible, meaning Layers X Heads X Layers X Heads of correlations.
        If display = True, then show (Layers * Heads) X (Layers * Heads) matrix.

        sample - to run the models on.
        sample2 - if provided the second model runs on this sample.
        attention_model1, attention_model2 - if provided, the func won't run the models.
        display - whether to display or just return the data.
        """

        if not attention_model1 or not attention_model2:
            attention_model1, attention_model2 = self.get_attentions(sample, sample2)
        data = self.comparator.compare_matrices(attention_model1, attention_model2)

        if display:
            #TODO: Move it from here, that's here just for now.
            data_flat = data.reshape(data.shape[0] * data.shape[1], data.shape[2] * data.shape[3])
            df_data = pd.DataFrame(data_flat)
            plt.figure(figsize=(40, 40))
            sns.heatmap(df_data, annot=True, cmap='coolwarm', vmin=0, vmax=1,
                             annot_kws={"fontsize": 4, "fontweight": 'bold'})
            title = f'Correlation between the attention matrices for the text:\n ' \
                    f'{sample["text"]}\n' \
                    f'Using the models: 1.{self.extractor1.model_name}. 2.{self.extractor2.model_name}'
            # add the title (wrap it doesn't get cut off)
            plt.title(title, fontsize=14, fontweight='bold', wrap=True)

            plt.xlabel('All attentions Model 1')
            plt.ylabel('All attentions Model 2')
            # Show the plot
            plt.show()

        return data

    def get_all_data_head_to_head_sample(self, sample: dict, sample2: dict = None,
                                         attention_model1 = None, attention_model2 = None, display: bool = True):

        """
        Gets a Layers X Heads correlation where each entry is the correlation of head j in layer i of both models.

        sample - to run the models on.
        sample2 - if provided the second model runs on this sample.
        attention_model1, attention_model2 - if provided, the func won't run the models.
        display - whether to display or just return the data.
        """

        if not attention_model1 or not attention_model2:
            attention_model1, attention_model2 = self.get_attentions(sample, sample2)

        data = self.comparator.compare_head_to_head(attention_model1, attention_model2)
        if display:
            self.comparator.plot_correlation_result_hitmap(sample['text'], self.extractor1.model_name,
                                                           self.extractor2.model_name, data)
        return data

    def hist_of_all_data(self, data):
        """
        Gets data of a sample of a shape (L,H,L,H)
        shows histogrm of the correlation of layer to layer compare to other heads.
        print the mean of those groups.
        """
        group1 = data[np.arange(data.shape[0]),:,np.arange(data.shape[0]),:].flatten()
        mask = np.ones_like(data)
        mask[np.arange(data.shape[0]),:,np.arange(data.shape[0]),:] = 0
        group2 = data[mask.astype(bool)]

        min_data = data.min()
        max_data = data.max()
        bins = np.linspace(min_data, max_data, 50)

        plt.hist(group1, bins, alpha=0.5, label='Layer to Layer corr')
        plt.hist(group2, bins, alpha=0.5, label='Other heads')
        plt.legend(loc='upper right')
        plt.show()

        print(f"Layer to Layer mean corr:{group1.mean()}")
        print(f"Other heads mean corr:{group2.mean()}")

if __name__ == '__main__':
    dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", 'clean', split='validation')
    sample = dataset[1]

    model1_dict = dict(model_name="bert-base-uncased", type='text')
    model2_dict = dict(model_name="facebook/wav2vec2-base-960h", type='audio')

    stats_generator = StatsGenerator(model1_dict, model2_dict, metric='Cosine')
    data_all = stats_generator.get_all_data_from_sample(sample)
    stats_generator.hist_of_all_data(data_all)

    sample['text'] = 'Hello, my dog is cute'
    sample2 = dataset[3]
    sample2['text'] = 'Bey, I am going home'

    model1_dict = dict(model_name="bert-base-uncased", type='text')
    model2_dict = dict(model_name="bert-base-uncased", type='text')

    stats_generator = StatsGenerator(model1_dict, model2_dict, metric='Cosine')
    data_all = stats_generator.get_all_data_from_sample(sample=sample, sample2=sample2)
    stats_generator.hist_of_all_data(data_all)

    sample2['text'] = 'Hey, your cat is ugly'
    data_all = stats_generator.get_all_data_from_sample(sample=sample, sample2=sample2)
    stats_generator.hist_of_all_data(data_all)




