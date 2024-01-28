from dataclasses import dataclass
from typing import Union, List

from .config import Config


@dataclass
class Sequence:
    start_frame: int
    end_frame: int
    style_start_frame: Union[None, int]
    style_end_frame: Union[None, int]


def config_to_sequences(a: Config) -> List[Sequence]:
    """
    Compares style frame indexes with image frame indexes to determine sequences.
    """
    acc = []
    for b in a.style_frames:
        acc.append(
            Sequence(
                start_frame = 0,
                end_frame = 0,
                style_start_frame = None,
                style_end_frame = None,
            )
        )

    return acc
