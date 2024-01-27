import os
import re
from dataclasses import dataclass
from typing import List, Literal

import cv2
import numpy

from ezsynth.utils.guides.guides import Guides, create_guides
from ezsynth.utils.sequences import Sequence, SequenceManager


@dataclass
class Config:
    styles: List[tuple[int, numpy.ndarray]]
    images: List[tuple[int, numpy.ndarray]]
    edge_method: Literal["PAGE", "PST", "Classic"]
    flow_method: Literal["RAFT", "DeepFlow"]
    model_name: Literal["sintel", "kitti", "chairs"]


def setup(
        style_path: str = "styles",
        input_path: str = "input",
        edge_method: Literal["PAGE", "PST", "Classic"] = "PAGE",
        flow_method: Literal["RAFT", "DeepFlow"] = "RAFT",
        model_name: Literal["sintel", "kitti", "chairs"] = "sintel"
) -> tuple[Config, Guides, List[Sequence]]:
    config = Config(
        _read_images(_get_image_paths(style_path)),
        _read_images(_get_image_paths(input_path)),
        edge_method,
        flow_method,
        model_name,
    )

    return config, create_guides(config), SequenceManager(config)._set_sequence()


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


def _read_images(a: List[tuple[int, str]]) -> List[tuple[int, numpy.ndarray]]:
    try:
        return [(i, cv2.imread(b)) for i, b in a]
    except Exception as e:
        raise ValueError(f"Error reading image: {e}")
