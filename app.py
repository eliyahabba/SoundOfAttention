import pandas as pd
import streamlit as st
from Common.Constants import Constants
from CorrelationAnalysis import CorrelationAnalysis
from StatsGenerator import StatsGenerator
from datasets import load_dataset
from transformers import BertTokenizer
import numpy as np
import plotly.express as px

AlignmentConstants = Constants.AlignmentConstants


@st.cache_resource
def get_stats_generator(metric_name: str):
    model1_dict = dict(model_name="bert-base-uncased", type='text')
    model2_dict = dict(model_name="facebook/wav2vec2-base-960h", type='audio')
    stats = StatsGenerator(model1_dict, model2_dict, metric=metric_name)
    return stats


@st.cache_resource
def get_resources():
    return load_dataset("patrickvonplaten/librispeech_asr_dummy", 'clean', split='validation'), \
        BertTokenizer.from_pretrained('bert-base-uncased'), \
        pd.read_pickle('data.pkl')


if __name__ == '__main__':
    st.set_page_config(layout="wide")

    dataset, tokenizer, attention_data = get_resources()

    st.title("Sound Of Attention")

    with st.sidebar:
        st.text("dataset: Librispeech/train")
        i = st.number_input("select index in dataset", 0, len(dataset)-1)
        metric_name = st.selectbox("select metric", ('KL', 'JS', 'Cosine', 'tot_var', 'pearson'), index=2)

    stats = get_stats_generator(metric_name)
    metric = CorrelationAnalysis(metric_name)

    st.subheader(f"Sample {i}")

    sample = dataset[i]
    st.audio(sample['audio']['array'], sample_rate=AlignmentConstants.FS)
    st.markdown(f"**text**: {sample['text'].lower()}")
    tokens = tokenizer.tokenize(sample['text'].lower())
    st.markdown(f'**tokens**: {" | ".join(tokens)}')

    st.subheader("Attention Visualization")
    st.markdown("**average by layer**: average attention weights of all heads in a layer")

    avg_by_layer_model1 = attention_data[sample['id']]['avg_by_layer_model1']
    avg_by_layer_model2 = attention_data[sample['id']]['avg_by_layer_model2']

    avg_layers_cmp = stats.comparator.compare_matrices(avg_by_layer_model1[:, None],
                                                       avg_by_layer_model2[:, None]).squeeze()
    sorted_indices = np.argsort(avg_layers_cmp, axis=None, )
    sorted_correlations = [dict(bert_layer=idx // avg_layers_cmp.shape[1],
                                wav2vec2_layer=idx % avg_layers_cmp.shape[1],
                                correlation=avg_layers_cmp[
                                    idx // avg_layers_cmp.shape[1], idx % avg_layers_cmp.shape[1]])
                           for idx in sorted_indices]
    high_correlation_first = st.checkbox("high correlation first", value=True)
    indices = st.selectbox("select layer comparison", sorted_correlations[::-1 if high_correlation_first else 1],
                           format_func=lambda x: f"[BERT] {x['bert_layer']} - [Wav2Vec2] {x['wav2vec2_layer']}: "
                                                 f"{x['correlation']:.3f}")

    cols = st.columns((2, 1, 2))
    with cols[0]:
        bert_layer_idx = st.number_input("select layer (bert)", 0,
                                         avg_by_layer_model1.shape[0] - 1,
                                         value=indices['bert_layer'])
    with cols[2]:
        wav2vec2_layer_idx = st.number_input("select layer (wav2vec2)", 0,
                                             avg_by_layer_model2.shape[0] - 1,
                                             value=indices['wav2vec2_layer'])

    correlation = metric.forward(avg_by_layer_model1[bert_layer_idx],
                                 avg_by_layer_model2[wav2vec2_layer_idx])
    st.text(f"correlation: {correlation:.3f}")
    cols = st.columns(2)
    with cols[0]:
        st.plotly_chart(px.imshow(avg_by_layer_model1[bert_layer_idx], x=tokens, y=tokens,
                                  color_continuous_scale='Blues', title="BERT"))
    with cols[1]:
        st.plotly_chart(px.imshow(avg_by_layer_model2[wav2vec2_layer_idx], x=tokens, y=tokens,
                                  color_continuous_scale='Blues', title="Wav2Vec2"))

    with st.expander("show all heads in layer"):
        if st.button("Generate data"):
            attention_lm, attention_asr = stats.get_attentions(sample, )
            cols = st.columns(2)
            with cols[0]:
                for head_idx in range(attention_lm.attentions.shape[1]):
                    st.plotly_chart(px.imshow(attention_lm.attentions[bert_layer_idx][head_idx], x=tokens, y=tokens,
                                              color_continuous_scale='Blues', title=f"Bert - head {head_idx}"))
            with cols[1]:
                for head_idx in range(attention_asr.attentions.shape[1]):
                    st.plotly_chart(px.imshow(attention_asr.attentions[wav2vec2_layer_idx][head_idx], x=tokens, y=tokens,
                                              color_continuous_scale='Blues', title=f"Wav2Vec2 - head {head_idx}"))

    st.subheader("Attention Correlation Stats")
    st.plotly_chart(px.imshow(avg_layers_cmp, labels={'x': 'Wav2Vec2', 'y': 'BERT'},
                              title="Correlation between all layers (avg)"))

