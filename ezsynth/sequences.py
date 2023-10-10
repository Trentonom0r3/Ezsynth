import numpy as np
import cv2
import math

class Sequence:
    """
    Helper class to store sequence information.

    :param begFrame: Index of the first frame in the sequence.
    :param endFrame: Index of the last frame in the sequence.
    :param keyframeIdx: Index of the keyframe in the sequence.
    :param style_image: Style image for the sequence.

    :return: Sequence object.

    """

    def __init__(self, begFrame, endFrame, style_start=None, style_end=None):
        self.begFrame = begFrame
        self.endFrame = endFrame
        self.style_start = style_start if style_start else None
        self.style_end = style_end if style_end else None
        if self.style_start is None and self.style_end is None:
                raise ValueError("At least one style attribute should be provided.")
       
    def __str__(self):
        return f"Sequence: {self.begFrame} - {self.endFrame} | {self.style_start} - {self.style_end}"
         
