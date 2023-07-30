import os.path
from typing import List
from datasets import load_dataset
from AttentionExtractors.ExtractorFactory import ExtractorFactory, ModelMetadata, DataType
from ForcedAlignment.AudioTextAttentionsMatcher import AudioTextAttentionsMatcher
from DataModels.Sample import Sample
import collections
import numpy as np
from tqdm import tqdm
import pandas as pd
import re


def get_dataset():
    dataset_name = 'albertvillanova/universal_dependencies'
    dataset_split = 'en_gum'
    dataset = load_dataset(dataset_name, dataset_split, split='validation')
    dataset = dataset.select_columns(['idx', 'text', 'tokens', 'head', 'deprel'])
    # change head values from str to int
    dataset = dataset.map(lambda example: {'head': [int(i) if i != 'None' else None for i in example['head']]})
    return dataset


def clean_dataset(dataset):
    def is_valid_sample(sample) -> bool:
        # filter non english sentences
        letters = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\' ')
        # numbers = set('0123456789')
        punctuations = set('!"(),-.:;[]–“”—?’')
        if set(sample['text']) - (letters | punctuations) != set():
            return False
        return True

    def clean_tokens(sample):
        def clean_token(token):
            # remove characters: ., (, )
            token = re.sub(r'[.()]', '', token)
            # replace ’ with '
            token = token.replace('’', '\'')
            # replace - with space
            token = token.replace('-', ' ')
            return token

        sample_df = pd.DataFrame.from_dict(dict(tokens=sample['tokens'], head=sample['head'], deprel=sample['deprel']))
        sample_df.index += 1  # change indices to start from 1, because 0 is root and non-word

        # remove tokens with head out-of-range
        assert (sample_df['head'] <= len(sample_df)).all()
        # remove tokens with deprel == 'punct',
        # and change the indices in "heads" list along with the "deprel" list accordingly
        punct_tokens = sample_df['deprel'] == 'punct'
        tokens_shift = punct_tokens.cumsum().astype(int)
        punct_indices = sample_df[punct_tokens].index
        sample_df.loc[sample_df['head'].isin(punct_indices), ['head', 'deprel']] = -1, None
        indices_to_shift = (~punct_tokens) & (sample_df['head'] >= 1)
        sample_df.loc[indices_to_shift, 'head'] = sample_df[indices_to_shift].apply(
            lambda x: x['head'] - tokens_shift[x['head']], axis=1)
        sample_df = sample_df.drop(punct_indices).reset_index(drop=True)
        # clean tokens text
        sample_df['tokens'] = sample_df['tokens'].apply(clean_token)

        sample['tokens'] = list(sample_df['tokens'])
        sample['head'] = list(sample_df['head'])
        sample['deprel'] = list(sample_df['deprel'])

        # clean sample['text'] from punctuations before space
        sample['text'] = re.sub(r'[^\w\s]+(\s|$)', ' ', sample['text'])
        sample['text'] = re.sub(r'(\s|^)[^\w\s]+', ' ', sample['text'])
        sample['text'] = clean_token(sample['text'])
        # space colapse
        sample['text'] = re.sub(r'\s+', ' ', sample['text']).strip()
        return sample

    dataset = dataset.filter(lambda example: is_valid_sample(example))
    dataset = dataset.map(clean_tokens)
    return dataset


class AttentionAsDependencySyntax:
    def __init__(self,
                 dependency_dataset,
                 lm_model_name: str = 'bert-base-uncased',
                 ):
        self.lm_model_name = lm_model_name
        model_metadata = ModelMetadata(model_name=self.lm_model_name, data_type=DataType.Text,
                                       align_tokens_to_bert_tokens=False, use_cls_and_sep=True)
        self.extractor = ExtractorFactory.create_attention_extractor(model_metadata)
        self.dependency_dataset = dependency_dataset
        self.dev_data = None

    def prepare_sample_for_extractor(self, item: int):
        sample = self.dependency_dataset[item]
        return Sample(id=sample['idx'], text=sample['text'])
        # " ".join(sample['tokens']))

    def get_attention(self, sample: Sample, i):
        if not os.path.exists(f'/home/vpnuser/cs_huji/anlp/ud_w2v2_attn_agg_tokens/{i}_{sample.id}.pkl'):
            print(f"skip {i} - not exists")
            return None
        attn = self.extractor.extract_attention(sample)
        audio_attn = pd.read_pickle(f'/home/vpnuser/cs_huji/anlp/ud_w2v2_attn_agg/{i}_{sample.id}.pkl')
        if attn.shape != audio_attn.shape:
            print(f"skip {i} - shape not equal")
            return None

        pd.to_pickle(attn, f'/home/vpnuser/cs_huji/anlp/ud_bert_attn/{i}_{sample.id}.pkl')

        return attn

    def get_aggregated_attention(self, attention, sample_tokens: List[str]):
        tokens_per_word = [self.extractor.text_model_processor.text_model.tokenizer.tokenize(token) for token in
                           sample_tokens]
        idx = 1
        matches = list()
        for word in tokens_per_word:
            matches.append(dict(audio_start=idx, audio_end=idx + len(word)))
            idx += len(word)
        return AudioTextAttentionsMatcher.group_attention_matrix_by_matches(attention, matches, use_cls_and_sep=True)

    def get_data_for_dependency_parser(self, item: int):
        labeled_sample = self.dependency_dataset[item]
        sample = self.prepare_sample_for_extractor(item)
        attention = self.get_attention(sample, item)
        if attention is None:
            return None
        aggregated_attention = self.get_aggregated_attention(attention, labeled_sample['tokens'])
        return dict(words=labeled_sample['tokens'],
                    heads=labeled_sample['head'],
                    relns=labeled_sample['deprel'],
                    attns=aggregated_attention)

    def _init_dev_data(self):
        if self.dev_data is None:
            self.dev_data = [self.get_data_for_dependency_parser(i) for i in tqdm(range(len(self.dependency_dataset)))]
            self.dev_data = [d for d in self.dev_data if d is not None]
            print(f"after remove none: {len(self.dev_data)}")
        return self.dev_data

    # Code for evaluating individual attention maps and baselines (Section 4.2 in paper)
    # - taken from attention-analysis/Syntax_Analysis.ipynb
    def evaluate_predictor(self, prediction_fn):
        """Compute accuracies for each relation for the given predictor."""
        self._init_dev_data()
        n_correct, n_incorrect = collections.Counter(), collections.Counter()
        for example_idx, example in enumerate(self.dev_data):
            words = example["words"]
            predictions = prediction_fn(example)
            for i, (p, y, r) in enumerate(zip(predictions, example["heads"],
                                              example["relns"])):
                if y >= len(example['relns']) or y < 0:
                    # print('head is out of range', y, len(example['relns']))
                    continue
                if r == 'punct' or example['relns'][y] == 'punct':
                    # if y == p:
                    #     print("skipped correct punct")
                    continue
                is_correct = (p == y)
                if r == "poss" and p < len(words):
                    # Special case for poss (see discussion in Section 4.2)
                    if i < len(words) and words[i + 1] == "'s" or words[i + 1] == "s'":
                        is_correct = (predictions[i + 1] == y)
                if is_correct:
                    n_correct[r] += 1
                    n_correct["all"] += 1
                else:
                    n_incorrect[r] += 1
                    n_incorrect["all"] += 1

                # if r=='case' and is_correct:
                #     print(example_idx, example['words'])
        return {k: n_correct[k] / float(n_correct[k] + n_incorrect[k])
                for k in n_incorrect.keys()}

    def attn_head_predictor(self, layer, head, mode="normal"):
        """Assign each word the most-attended-to other word as its head."""

        def predict(example):
            attn = np.array(example["attns"][layer][head])
            if mode == "transpose":
                attn = attn.T
            elif mode == "both":
                attn += attn.T
            else:
                assert mode == "normal"
            # ignore attention to self and [CLS]/[SEP] tokens
            attn[range(attn.shape[0]), range(attn.shape[0])] = 0
            attn = attn[1:-1, 1:-1]
            return np.argmax(attn, axis=-1) + 1  # +1 because ROOT is at index 0

        return predict

    def offset_predictor(self, offset):
        """Simple baseline: assign each word the word a fixed offset from
        it (e.g., the word to its right) as its head."""

        def predict(example):
            return [max(0, min(i + offset + 1, len(example["words"])))
                    for i in range(len(example["words"]))]

        return predict

    def get_scores(self, mode="normal"):
        """Get the accuracies of every attention head."""
        scores = collections.defaultdict(dict)
        for layer in range(12):
            for head in range(12):
                scores[layer][head] = self.evaluate_predictor(
                    self.attn_head_predictor(layer, head, mode))
        return scores

    def get_summary(self):
        # attn_head_scores[direction][layer][head][dep_relation] = accuracy
        attn_head_scores = {
            "dep->head": self.get_scores("normal"),
            "head<-dep": self.get_scores("transpose")
        }
        # baseline_scores[offset][dep_relation] = accuracy
        baseline_scores = {
            i: self.evaluate_predictor(self.offset_predictor(i)) for i in range(-3, 3)
        }

        all_res = list()
        for k in attn_head_scores:
            for layer in attn_head_scores[k]:
                for head in attn_head_scores[k][layer]:
                    tmp = attn_head_scores[k][layer][head]
                    tmp.update(dict(mode='test', direction=k, layer=layer, head=head))
                    all_res.append(tmp)
        for k in baseline_scores:
            tmp = baseline_scores[k]
            tmp.update(dict(mode='baseline', offset=k))
            all_res.append(tmp)
        df = pd.DataFrame(all_res)
        print(df.groupby(['mode', ]).max().transpose())
        print(df.groupby(['mode', ]).max().transpose().dropna().mean())
        return df


if __name__ == '__main__':
    dataset = clean_dataset(get_dataset())
    analyzer = AttentionAsDependencySyntax(dependency_dataset=dataset)
    # analyzer.evaluate_predictor(
    #     analyzer.attn_head_predictor(7, 5, 'normal'))
    df = analyzer.get_summary()
