from dataclasses import dataclass
from typing import Union, List

import cv2

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
    num_styles = len(a.style_frames)

    if num_styles == 1 and a.frames[0][0] == a.style_frames[0][0]:
        return [
            Sequence(start_frame = a.frames[0][0], end_frame = a.frames[-1][0], style_start = cv2.imread(styles[0]), style_end = None)
        ]

    if num_styles == 1 and a.frames[-1][0] == a.style_frames[0][0]:
        return [
            Sequence(start_frame = a.frames[0][0], end_frame = a.frames[-1][0], style_start = None, style_end = styles[0])
        ]

    sequences = []
    for i in range(num_styles - 1):

        # If both style indexes are not None
        if a.style_frames[i][0] is not None and a.style_frames[i + 1][0] is not None:
            if a.style_frames[i][0] == a.frames[0][0] and a.style_frames[i + 1][0] == a.frames[-1][0]:
                sequences.append(
                    Sequence(a.frames[0][0], a.frames[-1][0], styles[i], styles[i + 1]))

            # If the first style index is the first frame in the sequence
            elif a.style_frames[i][0] == a.frames[0][0] and a.style_frames[i + 1][0] != a.frames[-1][0]:
                sequences.append(Sequence(
                    a.frames[0][0], a.style_frames[i + 1][0], styles[i], styles[i + 1]))

            # If the second style index is the last frame in the sequence
            elif a.style_frames[i][0] != a.frames[0][0] and a.style_frames[i + 1][0] == a.frames[-1][0]:
                sequences.append(Sequence(
                    a.style_frames[i][0], a.frames[-1][0], styles[i], styles[i + 1]))

            elif a.style_frames[i][0] != a.frames[0][0] and a.style_frames[i + 1][0] != a.frames[-1][0] and a.style_frames[i][0] in imgindexes and a.style_frames[i + 1][0] in imgindexes:
                sequences.append(Sequence(
                    a.style_frames[i][0], a.style_frames[i + 1][0], styles[i], styles[i + 1]))

    return sequences
