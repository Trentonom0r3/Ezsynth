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
    Compares style frame indexes with image frame indexes to determine the sequences.
    """
    num_styles = len(a.style_frames)

    if num_styles == 1 and start_frame == style_indexes[0]:
        return [
            Sequence(start_frame = start_frame, end_frame = end_frame, style_start = cv2.imread(styles[0]))
        ]

    if style_indexes[0] == end_frame and num_styles == 1:
        return [
            Sequence(start_frame = start_frame, end_frame = end_frame, style_end = styles[0])
        ]

    sequences = []
    for i in range(num_styles - 1):

        # If both style indexes are not None
        if style_indexes[i] is not None and style_indexes[i + 1] is not None:
            if style_indexes[i] == start_frame and style_indexes[i + 1] == end_frame:
                sequences.append(
                    Sequence(start_frame, end_frame, styles[i], styles[i + 1]))

            # If the first style index is the first frame in the sequence
            elif style_indexes[i] == start_frame and style_indexes[i + 1] != end_frame:
                sequences.append(Sequence(
                    start_frame, style_indexes[i + 1], styles[i], styles[i + 1]))

            # If the second style index is the last frame in the sequence
            elif style_indexes[i] != start_frame and style_indexes[i + 1] == end_frame:
                sequences.append(Sequence(
                    style_indexes[i], end_frame, styles[i], styles[i + 1]))

            elif style_indexes[i] != start_frame and style_indexes[i + 1] != end_frame and style_indexes[i] in imgindexes and style_indexes[i + 1] in imgindexes:
                sequences.append(Sequence(
                    style_indexes[i], style_indexes[i + 1], styles[i], styles[i + 1]))

    return sequences
