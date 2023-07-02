from dataclasses import dataclass
from typing import Union


@dataclass
class Sample:
    text: Union[str, dict]
    audio: dict = None
