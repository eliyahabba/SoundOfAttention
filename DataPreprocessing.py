from Common.Constants import Constants
from Common.Resources import BasicResources
from ForcedAlignment.TextAudioMatcher import TextAudioMatcher

AudioConstants = Constants.AudioConstants


class DataPreprocessing:
    def __init__(self):
        self.resources = BasicResources()
        self.matcher = TextAudioMatcher(self.resources)

    def align_audio_text(self, audio_file, transcript_file):
        matches = self.matcher.match(transcript_file, audio_file)
        return matches

    def aggregate_attention_matrices(self, asr_attention_matrices):
        # code for aggregating ASR attention matrices to NLP token units
        pass

    def process_audio_file(self, audio_file):
        # code for processing audio file
        pass
