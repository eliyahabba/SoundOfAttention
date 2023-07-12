import numpy as np
import pandas as pd
from tqdm import tqdm

from Apps.analysis_generator_demo import get_analysis_generator, get_resources
from DataModels.Sample import Sample


class CreateAttentionsData:
    @staticmethod
    def create_attentions_data():
        analysis_generator = get_analysis_generator()
        dataset, tokenizer, _ = get_resources()

        data = list()
        for i in tqdm(len(dataset)):
            sample1 = Sample(id=dataset[i]["id"], text=dataset[i]["text"], audio=dataset[i]["audio"])
            sample2 = sample1

            attention_lm, attention_asr = analysis_generator.get_attentions(sample1, sample2)
            avg_by_layer_model1 = np.mean(attention_lm.attentions, axis=1)
            avg_by_layer_model2 = np.mean(attention_asr.attentions, axis=1)
            data.append(dict(sample_idx=i,
                             avg_by_layer_model1=avg_by_layer_model1,
                             avg_by_layer_model2=avg_by_layer_model2))
        return data

    @staticmethod
    def save_attentions_data(data):
        pd.to_pickle(data, "attentions.pkl")


if __name__ == '__main__':
    data = createAttentionsData.create_attentions_data()
    createAttentionsData.save_attentions_data(data)
