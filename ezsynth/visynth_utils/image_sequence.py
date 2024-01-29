import os
from typing import List, Tuple, Dict

import numpy as np


def image_sequence_from_directory(path: str) -> Tuple[List[np.ndarray], Dict[int, np.ndarray], int]:
    return _read_images(_get_image_paths(path))


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


def _read_images(a: List[tuple[int, str]]) -> List[Frame]:
    try:
        return [Frame(i, cv2.imread(b)) for i, b in a]
    except Exception as e:
        raise ValueError(f"Error reading image: {e}")
