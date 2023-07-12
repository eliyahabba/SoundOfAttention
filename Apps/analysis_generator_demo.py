from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datasets import load_dataset, Dataset
from transformers import BertTokenizer

from AttentionsAnalysis.AnalysisGenerator import AnalysisGenerator
from Common.Constants import Constants
from CorrelationsAnalysis.CorrelationAnalysis import CorrelationAnalysis
from DataModels.DataType import DataType
from DataModels.ModelMetadata import ModelMetadata
from DataModels.Sample import Sample

AlignmentConstants = Constants.AlignmentConstants
ATTENTIONS_BASE_PATH = Path(__file__).parents[1] / 'Data'


@st.cache_resource
def get_analysis_generator(metric_name: str, use_cls_and_sep: bool = True) -> AnalysisGenerator:
    model1_metadata = ModelMetadata(model_name="bert-base-uncased", data_type=DataType.Text,
                                    align_tokens_to_bert_tokens=False, use_cls_and_sep=use_cls_and_sep)
    model2_metadata = ModelMetadata(model_name="facebook/wav2vec2-base-960h", data_type=DataType.Audio,
                                    align_tokens_to_bert_tokens=True, use_cls_and_sep=use_cls_and_sep)
    analysis_generator = AnalysisGenerator(model1_metadata, model2_metadata, metric=metric_name)
    return analysis_generator


@st.cache_resource
def get_resources(use_cls_and_sep: bool = False) -> Tuple[Dataset, BertTokenizer, dict]:
    attentions_path = ATTENTIONS_BASE_PATH / f"attentions_with_cls_and_sep.pkl" if use_cls_and_sep else ATTENTIONS_BASE_PATH / f"attentions_without_cls_and_sep.pkl"
    return load_dataset("patrickvonplaten/librispeech_asr_dummy", 'clean', split='validation'), \
        BertTokenizer.from_pretrained('bert-base-uncased'), \
        pd.read_pickle(attentions_path)


def main():
    st.set_page_config(layout="wide")
    with st.sidebar:
        use_cls_and_sep = st.checkbox("use [CLS] and [SEP]. Currently it's affects only if you show all heads in layer",
                                      value=False)
    dataset, tokenizer, attention_data = get_resources(use_cls_and_sep)
    dataset_indices = get_dataset_indices(dataset, attention_data)

    st.title("Sound Of Attention")

    with st.sidebar:
        i_sorted = show_dataset_info(dataset_indices)
        metric_name = st.selectbox("select metric", ('KL', 'JS', 'Cosine', 'tot_var', 'pearson'), index=2)
    analysis_generator = get_analysis_generator(metric_name, use_cls_and_sep=use_cls_and_sep)
    metric = CorrelationAnalysis(metric_name)

    i = select_sample(dataset_indices, i_sorted)

    st.subheader(f"Sample {i}")

    sample = Sample(id=dataset[i]["id"], text=dataset[i]["text"], audio=dataset[i]["audio"])
    display_sample(analysis_generator, metric, sample, tokenizer, attention_data, use_cls_and_sep)


def get_dataset_indices(dataset: Dataset, attention_data: dict) -> pd.DataFrame:
    dataset_indices = pd.DataFrame([dict(id=sample['id'], index_in_dataset=i,
                                         mean_correlation=attention_data[sample['id']]['mean_correlation'])
                                    for i, sample in enumerate(dataset)]). \
        sort_values('mean_correlation', ascending=False).reset_index(drop=True)
    return dataset_indices


def show_dataset_info(dataset_indices: pd.DataFrame) -> int:
    st.text("dataset: Librispeech/train")
    i_sorted = st.number_input("select index in sorted dataset", 0, len(dataset_indices) - 1)
    st.text(f"id: {dataset_indices.iloc[i_sorted]['id']}\n"
            f"correlation (cosine): {dataset_indices.iloc[i_sorted]['mean_correlation']:.3f}")
    return i_sorted


def select_sample(dataset_indices: pd.DataFrame, i_sorted: int) -> int:
    i = int(dataset_indices.iloc[i_sorted]['index_in_dataset'])
    return i


def display_attention_correlation_analysis_generator(avg_layers_cmp: np.ndarray) -> None:
    st.subheader("Attention Correlation analysis_generator")
    st.plotly_chart(px.imshow(avg_layers_cmp, labels={'x': 'Wav2Vec2', 'y': 'BERT'},
                              title="Correlation between all layers (avg)"))


def display_sample(analysis_generator: AnalysisGenerator, metric: CorrelationAnalysis, sample: Sample,
                   tokenizer: BertTokenizer, attention_data: dict, use_cls_and_sep: bool) -> None:
    st.audio(sample.audio['array'], sample_rate=AlignmentConstants.FS)
    st.markdown(f"**text**: {sample.text.lower()}")
    tokens = tokenizer.tokenize(sample.text.lower())
    st.markdown(f'**tokens**: {" | ".join(tokens)}')

    st.subheader("Attention Visualization")
    st.markdown("**average by layer**: average attention weights of all heads in a layer")

    avg_by_layer_model1 = attention_data[sample.id]['avg_by_layer_model1']
    avg_by_layer_model2 = attention_data[sample.id]['avg_by_layer_model2']

    avg_layers_cmp = get_avg_layer_comparison(analysis_generator, avg_by_layer_model1, avg_by_layer_model2)

    high_correlation_first = st.checkbox("high correlation first", value=True)
    indices = select_layer_comparison(avg_layers_cmp, high_correlation_first)

    bert_layer_idx, wav2vec2_layer_idx = select_layer_indices(avg_by_layer_model1, avg_by_layer_model2, indices)

    correlation = calculate_correlation(metric, avg_by_layer_model1, avg_by_layer_model2, bert_layer_idx,
                                        wav2vec2_layer_idx)
    st.text(f"correlation: {correlation:.3f}")

    display_attention_heatmaps(avg_by_layer_model1, avg_by_layer_model2, tokens, bert_layer_idx, wav2vec2_layer_idx)
    display_all_heads_attention_heatmaps(analysis_generator, sample, tokens, bert_layer_idx, wav2vec2_layer_idx,
                                         use_cls_and_sep)

    display_attention_correlation_analysis_generator(avg_layers_cmp)


def get_avg_layer_comparison(analysis_generator: AnalysisGenerator, avg_by_layer_model1: np.ndarray,
                             avg_by_layer_model2: np.ndarray) -> np.ndarray:
    avg_layers_cmp = analysis_generator.get_correlations_of_attentions(avg_by_layer_model1[:, None],
                                                                       avg_by_layer_model2[:, None]).squeeze()
    return avg_layers_cmp


def select_layer_comparison(avg_layers_cmp: np.ndarray, high_correlation_first: bool) -> dict:
    sorted_indices = np.argsort(avg_layers_cmp, axis=None)
    sorted_correlations = [dict(bert_layer=idx // avg_layers_cmp.shape[1],
                                wav2vec2_layer=idx % avg_layers_cmp.shape[1],
                                correlation=avg_layers_cmp[
                                    idx // avg_layers_cmp.shape[1], idx % avg_layers_cmp.shape[1]])
                           for idx in sorted_indices]
    indices = st.selectbox("select layer comparison", sorted_correlations[::-1 if high_correlation_first else 1],
                           format_func=lambda x: f"[BERT] {x['bert_layer']} - [Wav2Vec2] {x['wav2vec2_layer']}: "
                                                 f"{x['correlation']:.3f}")
    return indices


def select_layer_indices(avg_by_layer_model1: np.ndarray, avg_by_layer_model2: np.ndarray, indices: dict) -> Tuple[
    int, int]:
    cols = st.columns((2, 1, 2))
    with cols[0]:
        bert_layer_idx = st.number_input("select layer (bert)", 0,
                                         avg_by_layer_model1.shape[0] - 1,
                                         value=indices['bert_layer'])
    with cols[2]:
        wav2vec2_layer_idx = st.number_input("select layer (wav2vec2)", 0,
                                             avg_by_layer_model2.shape[0] - 1,
                                             value=indices['wav2vec2_layer'])
    return bert_layer_idx, wav2vec2_layer_idx


def calculate_correlation(metric: CorrelationAnalysis, avg_by_layer_model1: np.ndarray, avg_by_layer_model2: np.ndarray,
                          bert_layer_idx: int, wav2vec2_layer_idx: int) -> float:
    correlation = metric.forward(avg_by_layer_model1[bert_layer_idx],
                                 avg_by_layer_model2[wav2vec2_layer_idx])
    return correlation


def display_attention_heatmaps(avg_by_layer_model1: np.ndarray, avg_by_layer_model2: np.ndarray, tokens: List[str],
                               bert_layer_idx: int, wav2vec2_layer_idx: int) -> None:
    cols = st.columns(2)
    with cols[0]:
        st.plotly_chart(px.imshow(avg_by_layer_model1[bert_layer_idx], x=tokens, y=tokens,
                                  color_continuous_scale='Blues', title="BERT"))
    with cols[1]:
        st.plotly_chart(px.imshow(avg_by_layer_model2[wav2vec2_layer_idx], x=tokens, y=tokens,
                                  color_continuous_scale='Blues', title="Wav2Vec2"))


def display_all_heads_attention_heatmaps(analysis_generator: AnalysisGenerator, sample: Sample, tokens: List[str],
                                         bert_layer_idx: int, wav2vec2_layer_idx: int, use_cls_and_sep: bool) -> None:
    if use_cls_and_sep:
        tokens = ['[CLS]'] + tokens + ['[SEP]']
    with st.expander("show all heads in layer"):
        if st.button("Generate data"):
            attention_lm, attention_asr = analysis_generator.get_attentions(sample, sample)
            cols = st.columns(2)
            with cols[0]:
                for head_idx in range(attention_lm.attentions.shape[1]):
                    st.plotly_chart(px.imshow(attention_lm.attentions[bert_layer_idx][head_idx], x=tokens, y=tokens,
                                              color_continuous_scale='Blues', title=f"Bert - head {head_idx}"))
            with cols[1]:
                for head_idx in range(attention_asr.attentions.shape[1]):
                    st.plotly_chart(
                        px.imshow(attention_asr.attentions[wav2vec2_layer_idx][head_idx], x=tokens, y=tokens,
                                  color_continuous_scale='Blues', title=f"Wav2Vec2 - head {head_idx}"))


if __name__ == '__main__':
    main()
