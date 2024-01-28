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
    if num_styles == 1 and begFrame == style_indexes[0]:
        return [
            Sequence(begFrame = begFrame, endFrame = endFrame, style_start = cv2.imread(styles[0]))
        ]

    if style_indexes[0] == endFrame and num_styles == 1:
        return [
            Sequence(begFrame = begFrame, endFrame = endFrame, style_end = styles[0])
        ]

    sequences = []
    for i in range(num_styles - 1):

        # If both style indexes are not None
        if style_indexes[i] is not None and style_indexes[i + 1] is not None:
            if style_indexes[i] == begFrame and style_indexes[i + 1] == endFrame:
                sequences.append(
                    Sequence(begFrame, endFrame, styles[i], styles[i + 1]))

            # If the first style index is the first frame in the sequence
            elif style_indexes[i] == begFrame and style_indexes[i + 1] != endFrame:
                sequences.append(Sequence(
                    begFrame, style_indexes[i + 1], styles[i], styles[i + 1]))

            # If the second style index is the last frame in the sequence
            elif style_indexes[i] != begFrame and style_indexes[i + 1] == endFrame:
                sequences.append(Sequence(
                    style_indexes[i], endFrame, styles[i], styles[i + 1]))

            elif style_indexes[i] != begFrame and style_indexes[i + 1] != endFrame and style_indexes[i] in imgindexes and style_indexes[i + 1] in imgindexes:
                sequences.append(Sequence(
                    style_indexes[i], style_indexes[i + 1], styles[i], styles[i + 1]))

    return sequences
