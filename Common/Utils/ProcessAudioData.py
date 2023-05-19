import numpy as np


class ProcessAudioData:
    @staticmethod
    def get_audio_values(audio: dict, audio_key: str = "array") -> np.ndarray:
        if audio_key not in audio:
            raise ValueError(f"Audio key '{audio_key}' not found in the provided audio dictionary.")
        audio_values = audio[audio_key]
        return audio_values
