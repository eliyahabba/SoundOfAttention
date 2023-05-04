from TextModel import TextModel
from AudioModel import AudioModel
from DataPreprocessing import DataPreprocessing
from CorrelationAnalysis import CorrelationAnalysis

class Main:
    def __init__(self):
        self.text_model = TextModel()
        self.audio_model = AudioModel()
        self.data_preprocessing = DataPreprocessing()
        self.correlation_analysis = CorrelationAnalysis()

    def run(self):
        pass

if __name__ == '__main__':
    main = Main()
    main.run()
