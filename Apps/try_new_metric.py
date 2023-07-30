import numpy as np
import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from DependencySyntax.universal_dependencies import get_dataset, clean_dataset, AttentionAsDependencySyntax


st.set_page_config(layout="wide")

bert_base_path = "/home/vpnuser/cs_huji/anlp/ud_bert_attn"
w2v2_base_path = "/home/vpnuser/cs_huji/anlp/ud_w2v2_attn_agg_tokens"


@st.cache_data
def init():
    dataset = clean_dataset(get_dataset())
    dep_analyzer = AttentionAsDependencySyntax(dataset)
    bert_files = os.listdir(bert_base_path)
    bert_files.sort(key=lambda x: int(x.split("_")[0]))
    return dataset, dep_analyzer, bert_files


dataset, dep_analyzer, bert_files = init()


threshold_methods = {
    "mean+2std": lambda attn: attn > attn.mean()+2*attn.std(),
    "mean+std": lambda attn: attn > attn.mean()+attn.std(),
    "constant_0.1": lambda attn: attn > 0.1,
    "constant_0.2": lambda attn: attn > 0.2,
    "constant_0.5": lambda attn: attn > 0.5,
    "top_1%": lambda attn: attn > np.quantile(attn, 0.99),
    "top_3%": lambda attn: attn > np.quantile(attn, 0.97),
    "top_5%": lambda attn: attn > np.quantile(attn, 0.95),
    "top_10%": lambda attn: attn > np.quantile(attn, 0.9),
}


similarity_metrics = {
    'Jaccard': lambda x, y: len(x.intersection(y)) / len(x.union(y)),
    # 'Dice': lambda x, y: 2 * len(x.intersection(y)) / (len(x) + len(y)),
    # 'Cosine': lambda x, y: len(x.intersection(y)) / (len(x) * len(y)),
    # 'Overlap': lambda x, y: len(x.intersection(y)) / min(len(x), len(y)),
}


def threshold_mat_to_indices_set(threshold_mat):
    return set([(i, j) for i in range(threshold_mat.shape[0])
                for j in range(threshold_mat.shape[1]) if threshold_mat[i, j]])


@st.cache_data
def get_attn_matrices(sample_idx):
    bert_fname = os.path.join(bert_base_path, bert_files[sample_idx])
    w2v2_fname = os.path.join(w2v2_base_path, bert_files[sample_idx])

    print("aggregating attentions...")
    bert_attn = dep_analyzer.get_aggregated_attention(pd.read_pickle(bert_fname), sample_data['tokens'])
    w2v2_attn = dep_analyzer.get_aggregated_attention(pd.read_pickle(w2v2_fname), sample_data['tokens'])
    print("done aggregating attentions")
    return bert_attn, w2v2_attn


if __name__ == '__main__':

    sample_idx = st.sidebar.number_input("Sample index", 0, len(bert_files), 0)

    index_in_dataset = int(bert_files[sample_idx].split("_")[0])
    sample_data = dataset[index_in_dataset]
    st.write(f"Sample index in dataset: {index_in_dataset}")
    st.write(f"Text: {sample_data['text']}")
    sample_df = pd.DataFrame.from_dict(dict(tokens=sample_data['tokens'],
                                heads=sample_data['head'],
                                deprel=sample_data['deprel']))
    st.dataframe(sample_df)

    bert_attn, w2v2_attn = get_attn_matrices(sample_idx)

    # choose layer and head for bert and for w2v2
    bert_layer = st.sidebar.number_input("Bert layer", 0, 12, 0)
    bert_head = st.sidebar.number_input("Bert head", 0, 11, 0)
    w2v2_layer = st.sidebar.number_input("W2V2 layer", 0, 11, 0)
    w2v2_head = st.sidebar.number_input("W2V2 head", 0, 11, 0)

    rels = st.multiselect("Relations", ['advmod', 'aux', 'case', 'det', 'fixed'],
                          default=['advmod', 'aux', 'case', 'det', 'fixed'])
    sample_df_for_trace = sample_df.copy()[(sample_df.heads != -1) & (sample_df.deprel.isin(rels))]
    st.dataframe(sample_df_for_trace)
    trace = go.Scatter(x=sample_df_for_trace.index,
                       y=sample_df_for_trace.heads-1,
                       text=sample_df_for_trace.deprel,
                       mode='markers', marker=dict(color='red'))
    trace_transpose = go.Scatter(x=sample_df_for_trace.heads-1,
                       y=sample_df_for_trace.index,
                       text=sample_df_for_trace.deprel,
                       mode='markers', marker=dict(color='orange'))

    threshold_method = st.sidebar.selectbox("Threshold method", threshold_methods.keys())

    bert_th = threshold_methods[threshold_method](bert_attn[bert_layer][bert_head])
    w2v2_th = threshold_methods[threshold_method](w2v2_attn[w2v2_layer][w2v2_head])
    st.text(f"Similarity between bert and w2v2:")
    for k in similarity_metrics:
        similarity = similarity_metrics[k](threshold_mat_to_indices_set(bert_th[1:-1, 1:-1]),
                                           threshold_mat_to_indices_set(w2v2_th[1:-1, 1:-1]))
        st.text(f"{k}: {similarity}")

    st.text("Attentions after thresholding")
    bert_col, w2v2_col = st.columns(2)
    with bert_col:
        st.plotly_chart(px.imshow(bert_th[1:-1, 1:-1]).
                        add_traces([trace, trace_transpose]))
    with w2v2_col:
        st.plotly_chart(px.imshow(w2v2_th[1:-1,1:-1]).
                        add_traces([trace, trace_transpose]))

    # display bert and w2v2 attentions as heatmaps
    st.text("Attentions")
    bert_col, w2v2_col = st.columns(2)
    with bert_col:
        st.plotly_chart(px.imshow(bert_attn[bert_layer][bert_head][1:-1, 1:-1]).add_traces([trace, trace_transpose]))
    with w2v2_col:
        st.plotly_chart(px.imshow(w2v2_attn[w2v2_layer][w2v2_head][1:-1, 1:-1]).add_traces([trace, trace_transpose] ))
