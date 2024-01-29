from dataclasses import dataclass
from typing import List, Dict
from typing import Literal

import numpy as np
import torch


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
