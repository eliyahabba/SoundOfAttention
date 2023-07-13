from dataclasses import dataclass
from typing import Union


@dataclass
class Sample:
    """
    This class represents a sample to our project.
    Each sample has a text and can have an audio, but not necessarily.

    :param id: The id of the sample.
    :param text: The text of the sample.
    :param audio: (Optional).
    The audio of the sample. The audio is a dictionary with the following keys:
    'array' (np.ndarray), 'sampling_rate' (int) ect.

    """
    id: Union[str, dict]
    text: Union[str, dict]
    audio: dict = None
