import numpy as np
import cv2

class Segmenter:
    def __init__(self, style, guides=None, masks=None):
        print(f"{type(style)=}")
        print(f"{type(guides)=}")
        self.original_style = style.copy()  # Store the original style
        self.current_style = style  # Working copy of the style
        self.guides = guides
        self.masks = masks
        self.index = 0

    def _reset_style(self):
        # Reset the style to its original state
        self.current_style = self.original_style.copy()

    def _segment_style(self, mask):
        # Segment the style using the mask
        return cv2.bitwise_and(self.current_style, self.current_style, mask=mask)

    def _segment_guide(self, guide_img, mask):
        # Segment the guide using the mask
        mask = cv2.bitwise_not(mask)
        return cv2.bitwise_and(guide_img, guide_img, mask=mask)

    def _combine(self, segmented_style, segmented_guide):
        # Combine the segmented style and guide
        return cv2.add(segmented_style, segmented_guide)

    def __call__(self, guide, mask):
        self._reset_style()  # Reset the style
        segmented_style = self._segment_style(mask)
        segmented_guide = self._segment_guide(guide, mask)
        combined = self._combine(segmented_style, segmented_guide)
        return combined, segmented_guide

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.guides):
            self.index = 0  # Reset index for future iterations
            raise StopIteration
        guide = self.guides[self.index]
        mask = self.masks[self.index]
        self.index += 1
        return self(guide, mask)
