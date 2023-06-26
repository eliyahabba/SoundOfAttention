from app import get_stats_generator, get_resources
from tqdm import tqdm
import numpy as np
import pandas as pd

if __name__ == '__main__':
    stats = get_stats_generator()
    dataset, tokenizer = get_resources()

    data = list()
    for i, sample in tqdm(enumerate(dataset)):
        attention_lm, attention_asr = stats.get_attentions(sample)
        avg_by_layer_model1 = np.mean(attention_lm.attentions, axis=1)
        avg_by_layer_model2 = np.mean(attention_asr.attentions, axis=1)
        data.append(dict(sample_idx=i,
                         avg_by_layer_model1=avg_by_layer_model1,
                         avg_by_layer_model2=avg_by_layer_model2))
    pd.to_pickle(data, "data.pkl")
