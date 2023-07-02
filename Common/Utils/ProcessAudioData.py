import numpy as np

from Common.Constants import Constants

AudioModelProcessorConstants = Constants.AudioModelProcessorConstants
AUDIO_KEY = AudioModelProcessorConstants.LIBRISPEECH_AUDIO_KEY

class ProcessAudioData:
    @staticmethod
    def get_audio_values(audio: dict,
                         ) -> np.ndarray:
        if AUDIO_KEY not in audio:
            raise ValueError(f"Audio key '{AUDIO_KEY}' not found in the provided audio dictionary.")
        audio_values = audio[AUDIO_KEY]
        return audio_values
