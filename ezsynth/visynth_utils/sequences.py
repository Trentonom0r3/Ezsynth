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

    def add(start_frame, end_frame):
        if start_frame < end_frame:
            acc.append(Sequence(start_frame = start_frame, end_frame = end_frame))

    for i, _ in a.style_frames:
        add(
            0 if len(acc) == 0 else acc[-1].end_frame,
            i,
        )

    add(
        acc[-1].end_frame,
        len(a.frames) - 1,
    )

    return acc
