from transformers import Wav2Vec2Processor
from Constants import Constants
AudioConstants = Constants.AudioConstants

class DataPreprocessing:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(AudioConstants.W2V2)

    def align_audio_text(self, audio_file, transcript_file):
        # code for aligning audio and text
        pass

    def aggregate_attention_matrices(self, asr_attention_matrices):
        # code for aggregating ASR attention matrices to NLP token units
        pass

    def process_audio_file(self, audio_file):
        # code for processing audio file
        pass
