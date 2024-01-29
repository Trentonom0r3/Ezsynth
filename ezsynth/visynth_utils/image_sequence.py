import os
import re
from typing import List, Tuple

import cv2
import numpy as np


def image_sequence_from_directory(
        frames_directory: str,
        style_frames_directory: str,
) -> Tuple[List[np.ndarray], List[Tuple[int, np.ndarray]], int]:
    frames = _read_images(_get_image_paths(frames_directory))
    frame_offset = 0 if len(frames) == 0 else frames[0][0]
    style_frames = _read_images(_get_image_paths(style_frames_directory))
    return (
        [x for _, x in frames],
        [(i - frame_offset, x) for i, x in style_frames],
        frame_offset,
    )


def _get_image_paths(path: str) -> List[tuple[int, str]]:
    try:
        return sorted([
            (_extract_index(x), os.path.join(path, x)) for x in os.listdir(path)
        ])
    except Exception:
        raise ValueError("Cannot read images in: " + path)


def _extract_index(name: str):
    try:
        pattern = re.compile(r"(\d+)\.(jpg|jpeg|png)$")
        return int(pattern.findall(name)[0][0])
    except Exception:
        raise ValueError("Cannot extract index from: " + name)


def _read_images(a: List[tuple[int, str]]) -> List[Tuple[int, np.ndarray]]:
    try:
        return [(i, cv2.imread(b)) for i, b in a]
    except Exception as e:
        raise ValueError(f"Error reading image: {e}")
