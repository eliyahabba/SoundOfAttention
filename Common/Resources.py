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
    def __init__(self, text_model_name = 'bert-base-uncased'):
        self.text_model_name = text_model_name

    def load(self):
        tokenizer = BertTokenizer.from_pretrained(self.text_model_name)
        dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", 'clean', split='validation')
        aligner = Wav2Vec2Aligner("facebook/wav2vec2-base-960h", cuda=False)
        return tokenizer, dataset, aligner


class StreamlitResources(Resources):
    """
    Concrete class for loading resources for the TextAudioMatcher when using Streamlit.
    """
    def __init__(self, text_model_name = 'bert-base-uncased'):
        self.text_model_name = text_model_name

    @st.cache_resource
    def load(self):
        with st.spinner("Loading resources, please wait..."):
            tokenizer = BertTokenizer.from_pretrained(self.text_model_name)
            dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", 'clean', split='validation')
            aligner = Wav2Vec2Aligner("facebook/wav2vec2-base-960h", cuda=False)
        return tokenizer, dataset, aligner
