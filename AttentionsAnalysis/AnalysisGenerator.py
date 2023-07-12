from typing import Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

from AttentionExtractors.ExtractorFactory import ExtractorFactory
from AttentionsComparators.AttentionsComparator import AttentionsComparator
from Common.Constants import Constants
from CorrelationsAnalysis.CorrelationAnalysis import CorrelationAnalysis
from DataModels.Attentions import Attentions
from DataModels.DataType import DataType
from DataModels.ModelMetadata import ModelMetadata
from DataModels.Sample import Sample
from Visualizers.VisualizerAttentionsResults import VisualizerAttentionsResults

DEFAULT_AUDIO_KEY = Constants.AudioModelProcessorConstants.LIBRISPEECH_AUDIO_KEY
DISPLAY = True


class AnalysisGenerator:
    def __init__(self, model1_metadata: ModelMetadata, model2_metadata: ModelMetadata, metric: str = "pearson"):
        self.metric = metric
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.extractor1 = ExtractorFactory.create_attention_extractor(model1_metadata, device=self.device)
        self.extractor2 = ExtractorFactory.create_attention_extractor(model2_metadata, device=self.device)
        self.comparator = AttentionsComparator(correlation_analysis=CorrelationAnalysis(metric=metric))

    def get_attentions(self, sample1: Sample, sample2: Sample) -> Tuple[Attentions, Attentions]:
        """
        Runs the models on the sample and returns the attention matrices.
        :param sample1: sample to run the first model on.
        :param sample2: sample to run the second model on.
        :return: Attention matrices of the two models.
        """
        attention_model1 = self.extractor1.extract_attention(sample=sample1)
        attention_model2 = self.extractor2.extract_attention(sample=sample2)

        assert attention_model1.shape == attention_model2.shape

        return attention_model1, attention_model2

    def extract_layerwise_attention_averages(self, attention_model1: Attentions, attention_model2: Attentions) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Extracts and returns the average by layer of attentions for two attention models.

        :param attention_model1: Attention model of sample1.
        :param attention_model2: Attention model of sample2.
        :return: Tuple of two numpy arrays of shape (L,L) where L is the number of layers.
        """
        avg_by_layer_model1 = np.mean(attention_model1.attentions, axis=1)[:, None]
        avg_by_layer_model2 = np.mean(attention_model2.attentions, axis=1)[:, None]
        return avg_by_layer_model1, avg_by_layer_model2

    # def calculate_layerwise_correlation_matrix(self, avg_by_layer_model1: np.ndarray,
    #                                                            avg_by_layer_model2: np.ndarray) -> np.ndarray:
    #     correlations_comparisons = self.comparator.compare_attention_matrices(avg_by_layer_model1,
    #                                                                           avg_by_layer_model2).squeeze()
    #     return correlations_comparisons

    def calculate_layerwise_correlation_averages(self, attention_model1: Attentions, attention_model2: Attentions,
                                                 display: bool = True) -> np.ndarray:
        """
        Does average on all head of each layer and produce a Layers X Layers matrix of avg
        correlation between the layers.

        :param attention_model1: Attention model of sample1.
        :param attention_model2: Attention model of sample2.
        :param display: Whether to display or just return the data.
        :return: correlation matrix of shape (L,L)
        """

        # Mean all heads in the same layer. Shape [L, 1, tokens, tokens]
        avg_by_layer_model1, avg_by_layer_model2 = self.extract_layerwise_attention_averages(attention_model1,
                                                                                             attention_model2)

        correlations_comparisons = self.get_correlations_of_attentions(avg_by_layer_model1, avg_by_layer_model2)

        if display:
            VisualizerAttentionsResults.plot_correlation_of_attentions_by_avg_of_each_layer(
                sample=sample1,
                model_name1=self.extractor1.model_metadata.model_name,
                model_name2=self.extractor2.model_metadata.model_name,
                correlations_comparisons=correlations_comparisons)
        return correlations_comparisons

    def get_correlations_of_attentions(self, attention_model1: Attentions, attention_model2: Attentions) -> np.ndarray:
        """
        gets the whole data is possible, meaning Layers X Heads X Layers X Heads of correlations.
        If display = True, then show (Layers * Heads) X (Layers * Heads) matrix.

        :param attention_model1: Attention model of sample1.
        :param attention_model2: Attention model of sample2.

        :return: correlation matrix of shape (L,H,L,H)
        """

        full_correlations_comparisons = self.comparator.compare_attention_matrices(attention_model1, attention_model2)
        return full_correlations_comparisons

    def get_all_data_head_to_head_sample(self, attention_model1: Attentions,
                                         attention_model2: Attentions) -> pd.DataFrame:
        """
        Gets the data of a sample of a shape (L,H,L,H)
        :param attention_model1: Attention model of sample1.
        :param attention_model2: Attention model of sample2.
        :return: correlation matrix of shape (L,H,L,H)
        """

        head_to_head_correlations_comparisons = self.comparator.compare_head_to_head(attention_model1, attention_model2)
        return head_to_head_correlations_comparisons


if __name__ == '__main__':
    display = DISPLAY
    dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", 'clean', split='validation')

    # Example 1 - Compare text to audio
    # Start
    sample1 = Sample(id=dataset[1]["id"], text=dataset[1]["text"], audio=dataset[1]["audio"])
    sample2 = sample1

    model1_metadata = ModelMetadata(model_name="bert-base-uncased", data_type=DataType.Text,
                                    align_tokens_to_bert_tokens=False, use_cls_and_sep=True)
    model2_metadata = ModelMetadata(model_name="facebook/wav2vec2-base-960h", data_type=DataType.Audio,
                                    align_tokens_to_bert_tokens=True, use_cls_and_sep=True)

    analysis_generator = AnalysisGenerator(model1_metadata, model2_metadata, metric='Cosine')
    attention_model1, attention_model2 = analysis_generator.get_attentions(sample1, sample2)
    full_correlations_comparisons = analysis_generator.get_correlations_of_attentions(attention_model1, attention_model2)
    VisualizerAttentionsResults.plot_correlation_of_attentions(sample=sample1,
                                                               model_name1=model1_metadata.model_name,
                                                               model_name2=model2_metadata.model_name,
                                                               correlations_comparisons=full_correlations_comparisons)
    VisualizerAttentionsResults.plot_histogram_of_layers_and_heads(full_correlations_comparisons)
    # End

    # Example 2 - Compare text to text with CLS and SEP and then without
    # Start
    sample1.text = 'Hello, my dog is cute'
    sample2 = Sample(id=dataset[1]["id"], text=dataset[3]["text"], audio=dataset[3]["audio"])
    sample2.text = 'Bey, I am going home'

    model1_metadata = ModelMetadata(model_name="bert-base-uncased", data_type=DataType.Text,
                                    align_tokens_to_bert_tokens=False, use_cls_and_sep=True)
    model2_metadata = model1_metadata
    analysis_generator = AnalysisGenerator(model1_metadata, model2_metadata, metric='Cosine')
    attention_model1, attention_model2 = analysis_generator.get_attentions(sample1, sample2)
    full_correlations_comparisons2_with_cls = analysis_generator.get_correlations_of_attentions(attention_model1,
                                                                                             attention_model2)
    head_to_head_correlations_comparisons_with_cls = analysis_generator.get_all_data_head_to_head_sample(attention_model1,
                                                                                                      attention_model2)
    VisualizerAttentionsResults.plot_correlation_of_attentions_by_comparing_head_to_head(sample=sample1,
                                                                                         model_name1=model1_metadata.model_name,
                                                                                         model_name2=model2_metadata.model_name,
                                                                                         correlation_df=head_to_head_correlations_comparisons_with_cls)

    model1_metadata = ModelMetadata(model_name="bert-base-uncased", data_type=DataType.Text,
                                    align_tokens_to_bert_tokens=False, use_cls_and_sep=False)
    model2_metadata = model1_metadata
    analysis_generator = AnalysisGenerator(model1_metadata, model2_metadata, metric='Cosine')
    attention_model1, attention_model2 = analysis_generator.get_attentions(sample1, sample2)
    full_correlations_comparisons2_without_cls = analysis_generator.get_correlations_of_attentions(attention_model1,
                                                                                                attention_model2)
    head_to_head_correlations_comparisons2_without_cls = analysis_generator.get_all_data_head_to_head_sample(
        attention_model1,
        attention_model2)
    VisualizerAttentionsResults.plot_correlation_of_attentions_by_comparing_head_to_head(sample=sample1,
                                                                                         model_name1=model1_metadata.model_name,
                                                                                         model_name2=model2_metadata.model_name,
                                                                                         correlation_df=head_to_head_correlations_comparisons2_without_cls)

    # End

    # Example 3 - Compare audio to audio
    # Start
    # sample2.text = 'Hey, your cat is ugly'
    # analysis_generator = AnalysisGenerator(model1_metadata, model2_metadata, metric='Cosine')
    # attention_model1, attention_model2 = analysis_generator.get_attentions(sample1, sample2)
    # full_correlations_comparisons4 = analysis_generator.get_correlations_of_attentions(attention_model1, attention_model2)
    # # End

    # Example 4 - Compare text to text with CLS and SEP and then without - Roberta
    # Start
    sample1.text = 'Hello, my dog is cute'
    sample2 = Sample(id=dataset[3]["id"], text=dataset[3]["text"], audio=dataset[3]["audio"])
    sample2.text = 'Bey, I am going home'

    model1_metadata = ModelMetadata(model_name="roberta-base", data_type=DataType.Text,
                                    align_tokens_to_bert_tokens=False, use_cls_and_sep=True)
    model2_metadata = model1_metadata
    analysis_generator = AnalysisGenerator(model1_metadata, model2_metadata, metric='Cosine')
    attention_model1, attention_model2 = analysis_generator.get_attentions(sample1, sample2)
    full_correlations_comparisons3_with_cls = analysis_generator.get_correlations_of_attentions(attention_model1,
                                                                                             attention_model2)
    head_to_head_correlations_comparisons3_with_cls = analysis_generator.get_all_data_head_to_head_sample(attention_model1,
                                                                                                       attention_model2)

    model1_metadata = ModelMetadata(model_name="roberta-base", data_type=DataType.Text,
                                    align_tokens_to_bert_tokens=False, use_cls_and_sep=False)
    model2_metadata = model1_metadata
    analysis_generator = AnalysisGenerator(model1_metadata, model2_metadata, metric='Cosine')
    attention_model1, attention_model2 = analysis_generator.get_attentions(sample1, sample2)
    full_correlations_comparisons3_without_cls = analysis_generator.get_correlations_of_attentions(attention_model1,
                                                                                                attention_model2)
    head_to_head_correlations_comparisons3_without_cls = analysis_generator.get_all_data_head_to_head_sample(
        attention_model1,
        attention_model2)
    # End

    # Example 4 - Compare text to text with CLS and SEP and then without - Roberta & Bert
    # Start
    sample1.text = 'Hello, my dog is cute'
    sample2 = Sample(id=dataset[3]["id"], text=dataset[3]["text"], audio=dataset[3]["audio"])
    sample2.text = 'Hello, my dog is cute'

    model1_metadata = ModelMetadata(model_name="roberta-base", data_type=DataType.Text,
                                    align_tokens_to_bert_tokens=False, use_cls_and_sep=True)
    model2_metadata = ModelMetadata(model_name="bert-base-uncased", data_type=DataType.Text,
                                    align_tokens_to_bert_tokens=False, use_cls_and_sep=True)
    analysis_generator = AnalysisGenerator(model1_metadata, model2_metadata, metric='Cosine')
    attention_model1, attention_model2 = analysis_generator.get_attentions(sample1, sample2)
    full_correlations_comparisons4_with_cls = analysis_generator.get_correlations_of_attentions(attention_model1,
                                                                                             attention_model2)
    head_to_head_correlations_comparisons4_with_cls = analysis_generator.get_all_data_head_to_head_sample(attention_model1,
                                                                                                       attention_model2)

    model1_metadata = ModelMetadata(model_name="roberta-base", data_type=DataType.Text,
                                    align_tokens_to_bert_tokens=False, use_cls_and_sep=False)
    model2_metadata = ModelMetadata(model_name="bert-base-uncased", data_type=DataType.Text,
                                    align_tokens_to_bert_tokens=False, use_cls_and_sep=False)
    analysis_generator = AnalysisGenerator(model1_metadata, model2_metadata, metric='Cosine')
    attention_model1, attention_model2 = analysis_generator.get_attentions(sample1, sample2)
    full_correlations_comparisons4_without_cls = analysis_generator.get_correlations_of_attentions(attention_model1,
                                                                                                attention_model2)
    head_to_head_correlations_comparisons4_without_cls = analysis_generator.get_all_data_head_to_head_sample(
        attention_model1,
        attention_model2)
    # End
