import streamlit as st
from datasets import load_dataset
from transformers import BertTokenizer

from Wav2Vec2Aligner import Wav2Vec2Aligner
from Common.Constants import Constants
TextConstants = Constants.TextConstants
FS = 16000


class Resources:
    @staticmethod
    @st.cache_resource
    def load():
        with st.spinner("Loading resources, please wait..."):
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", 'clean', split='validation')
            aligner = Wav2Vec2Aligner("facebook/wav2vec2-base-960h", cuda=False)
        return tokenizer, dataset, aligner


class Demo:
    def __init__(self):
        self.resources = Resources()
        self.tokenizer, self.dataset, self.aligner = self.resources.load()

    def run(self):
        st.title("Bert-Audio Demo")
        st.subheader("Librispeech dummy dataset")

        i = st.number_input("index in dataset", 0, len(self.dataset))
        sample = self.dataset[i]
        self._display_sample(sample)

    def _display_sample(self, sample):
        sample_text = sample['text']
        assert sample['audio']['sampling_rate'] == FS
        assert "' " not in sample['text'], f"skipping examples with '<space>' due to tokenizer issues"
        st.audio(sample['audio']['array'], sample_rate=FS)
        st.markdown(sample_text.lower())

        sample_tokens = self.tokenizer.tokenize(sample_text)
        assert self.tokenizer.convert_tokens_to_string(sample_tokens).replace(" ' ",
                                                                              "'") == sample_text.lower()

        segments = self.aligner(sample['audio']['array'], sample_text)
        tokens_mapping = self._create_tokens_mapping(sample_tokens, segments)

        for mapping in tokens_mapping:
            self._display_mapping(mapping, sample)

    def _create_tokens_mapping(self, sample_tokens, segments):
        tokens_mapping = list()
        seg_idx = -1
        for token in sample_tokens:
            if not token.startswith("#"):
                seg_idx += 1
            stripped_token = token.replace("#", "")
            segments_for_token = segments[seg_idx: seg_idx + len(stripped_token)]
            assert stripped_token == "".join([s.label for s in segments_for_token]).lower()
            seg_idx += len(stripped_token)
            tokens_mapping.append(dict(token=token,
                                       stripped_token=stripped_token,
                                       start=segments_for_token[0].start,
                                       end=segments_for_token[-1].end,
                                       score=sum([s.score for s in segments_for_token]) / len(segments_for_token)))
        return tokens_mapping

    def _display_mapping(self, mapping, sample):
        token_col, audio_col = st.columns([1, 4])
        with token_col:
            st.markdown(mapping["token"])
        with audio_col:
            st.audio(sample['audio']['array'][int(mapping['start'] * 16000 / 50): int(mapping['end'] * 16000 / 50)],
                     sample_rate=FS)


if __name__ == '__main__':
    demo = Demo()
    demo.run()
