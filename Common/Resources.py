from abc import ABC, abstractmethod

import streamlit as st
from datasets import load_dataset
from transformers import BertTokenizer, RobertaTokenizer

from Common.Constants import Constants
from ForcedAlignment.Wav2Vec2Aligner import Wav2Vec2Aligner

TextConstants = Constants.TextConstants


class Resources(ABC):
    """
    Abstract class for loading resources for the TextAudioMatcher.
    """

    @abstractmethod
    def load(self):
        pass


class BasicResources(Resources):
    """
    Concrete class for loading resources for the TextAudioMatcher when not using Streamlit.
    """

    def load(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", 'clean', split='validation')
        aligner = Wav2Vec2Aligner("facebook/wav2vec2-base-960h", cuda=False)
        return tokenizer, dataset, aligner


class StreamlitResources(Resources):
    """
    Concrete class for loading resources for the TextAudioMatcher when using Streamlit.
    """

    @staticmethod
    @st.cache_resource
    def load():
        with st.spinner("Loading resources, please wait..."):
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", 'clean', split='validation')
            aligner = Wav2Vec2Aligner("facebook/wav2vec2-base-960h", cuda=False)
        return tokenizer, dataset, aligner
