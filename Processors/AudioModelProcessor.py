import numpy as np
from transformers.modeling_outputs import Wav2Vec2BaseModelOutput

from Common.Constants import Constants
from DataModels.AudioModel import AudioModel

AudioModelProcessorConstants = Constants.AudioModelProcessorConstants


class AudioModelProcessor:
    """
        A class representing an audio processing model (e.g., Wav2Vec2)

        Parameters:
            audio_model (AudioModel): The audio model
    """

    def __init__(self, audio_model: AudioModel):
        self.audio_model = audio_model

    def run(self, audio_values: np.ndarray) -> Wav2Vec2BaseModelOutput:
        inputs = self.audio_model.processor(audio_values, sampling_rate=AudioModelProcessorConstants.SAMPLING_RATE,
                                            return_tensors="pt", padding=True)
        inputs = inputs.to(self.audio_model.device)
        outputs = self.audio_model(inputs.input_values)
        return outputs
