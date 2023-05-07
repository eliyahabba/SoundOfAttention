import streamlit as st

from Common.Constants import Constants
from Common.Resources import StreamlitResources
from ForcedAlignment.TextAudioMatcher import TextAudioMatcher

TextConstants = Constants.TextConstants
AlignmentConstants = Constants.AlignmentConstants


class Demo:
    def __init__(self):
        self.resources = StreamlitResources()
        self.matcher = TextAudioMatcher(self.resources)

    def run(self):
        st.title("Bert-Audio Demo")
        st.subheader("Librispeech dummy dataset")

        i = st.number_input("index in dataset", 0, len(self.matcher.dataset))
        sample = self.matcher.dataset[i]
        self._display_sample(sample)

    def _display_sample(self, sample):
        matches = self.matcher.match(sample['text'], sample['audio'])

        st.audio(sample['audio']['array'], sample_rate=AlignmentConstants.FS)
        st.markdown(sample['text'].lower())

        for match in matches:
            self._display_match(match)

    def _display_match(self, match):
        st.audio(match['audio'], sample_rate=AlignmentConstants.FS)
        st.markdown(match['text'].lower())


if __name__ == '__main__':
    demo = Demo()
    demo.run()
