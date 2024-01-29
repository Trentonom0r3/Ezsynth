import os
import re
from dataclasses import dataclass
from typing import List, Dict
from typing import Literal

import cv2
import numpy as np
import torch

from .frame import Frame


def auto_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        torch.device("cpu")


@dataclass
class Config:
    """
    Visynth config.
    :param frames: List of video images.
    :param style_frames: Dictionary of frame indexes and style video images.
    :param edge_method: Method for edge detection. PAGE, PST or Classic. Default is PAGE.
    :param flow_method: Method for optical flow computation. RAFT or DeepFlow. Default is RAFT.
    :param flow_model: Model name for optical flow. sintel, kitti or chairs. Default is sintel.
    :param device: What processing unit to use.
    """
    frames: List[np.ndarray]
    style_frames: Dict[int, np.ndarray]
    edge_method: Literal["PAGE", "PST", "classic"] = "PAGE"
    flow_method: Literal["RAFT", "DeepFlow"] = "RAFT"
    flow_model: Literal["sintel", "kitti", "chairs"] = "sintel"
    device: torch.device = auto_device()


def image_sequence_from_directory(path: str) -> List[Frame]:
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
