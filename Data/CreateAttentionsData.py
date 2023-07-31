import numpy as np
import pandas as pd
from tqdm import tqdm

from Apps.analysis_generator_demo import get_analysis_generator, get_resources
from DataModels.Sample import Sample


class CreateAttentionsData:
    @staticmethod
    def create_attentions_data(metric_name="Cosine", use_cls_and_sep=True):
        analysis_generator = get_analysis_generator(metric_name=metric_name, use_cls_and_sep=use_cls_and_sep)
        dataset, tokenizer, _ = get_resources()

        data = list()
        for i in tqdm(range(len(dataset))):
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
    def save_attentions_data(data, metric_name="Cosine", use_cls_and_sep=True):
        use_special_tokens = "with_cls_and_sep" if use_cls_and_sep else "without_cls_and_sep"
        pd.to_pickle(data, f'attentions_{use_special_tokens}_{metric_name}.pkl')


if __name__ == '__main__':
    data = CreateAttentionsData.create_attentions_data(metric_name="Cosine", use_cls_and_sep=True)
    CreateAttentionsData.save_attentions_data(data, metric_name="Cosine", use_cls_and_sep=True)
