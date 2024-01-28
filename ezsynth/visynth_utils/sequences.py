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
    sequences = []
    if self.num_styles == 1 and self.begFrame == self.style_indexes[0]:
        sequences.append(
            Sequence(begFrame = self.begFrame, endFrame = self.endFrame, style_start = cv2.imread(self.styles[0])))

        return sequences

    if self.style_indexes[0] == self.endFrame and self.num_styles == 1:
        sequences.append(
            Sequence(begFrame = self.begFrame, endFrame = self.endFrame, style_end = self.styles[0]))

        return sequences

    for i in range(self.num_styles - 1):

        # If both style indexes are not None
        if self.style_indexes[i] is not None and self.style_indexes[i + 1] is not None:
            if self.style_indexes[i] == self.begFrame and self.style_indexes[i + 1] == self.endFrame:
                sequences.append(
                    Sequence(self.begFrame, self.endFrame, self.styles[i], self.styles[i + 1]))

            # If the first style index is the first frame in the sequence
            elif self.style_indexes[i] == self.begFrame and self.style_indexes[i + 1] != self.endFrame:
                sequences.append(Sequence(
                    self.begFrame, self.style_indexes[i + 1], self.styles[i], self.styles[i + 1]))

            # If the second style index is the last frame in the sequence
            elif self.style_indexes[i] != self.begFrame and self.style_indexes[i + 1] == self.endFrame:
                sequences.append(Sequence(
                    self.style_indexes[i], self.endFrame, self.styles[i], self.styles[i + 1]))

            elif self.style_indexes[i] != self.begFrame and self.style_indexes[i + 1] != self.endFrame and self.style_indexes[i] in self.imgindexes and self.style_indexes[i + 1] in self.imgindexes:
                sequences.append(Sequence(
                    self.style_indexes[i], self.style_indexes[i + 1], self.styles[i], self.styles[i + 1]))

    return sequences
