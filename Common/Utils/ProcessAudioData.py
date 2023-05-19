import numpy as np

from Common.Constants import Constants

AudioModelProcessorConstants = Constants.AudioModelProcessorConstants


class ProcessAudioData:
    @staticmethod
    def get_audio_values(audio: dict,
                         audio_key: str = AudioModelProcessorConstants.LIBRISPEECH_AUDIO_KEY) -> np.ndarray:
        if audio_key not in audio:
            raise ValueError(f"Audio key '{audio_key}' not found in the provided audio dictionary.")
        audio_values = audio[audio_key]
        return audio_values
