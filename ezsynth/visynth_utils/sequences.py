from dataclasses import dataclass
from typing import List

from .config import Config


@dataclass
class Sequence:
    start_frame: int
    end_frame: int


def config_to_sequences(a: Config) -> List[Sequence]:
    """
    Compares style frame indexes with image frame indexes to determine sequences.
    """
    acc = []
    for b in a.style_frames:
        acc.append(
            Sequence(
                start_frame = first_frame if len(acc) == 0 else acc[-1].end_frame,
                end_frame = b.index,
            )
        )

    return acc
