from dataclasses import dataclass
from typing import List, Literal

import numpy


@dataclass
class Config:
    styles: List[tuple[int, numpy.ndarray]]
    images: List[tuple[int, numpy.ndarray]]
    edge_method: Literal["PAGE", "PST", "Classic"]
    flow_method: Literal["RAFT", "DeepFlow"]
    model_name: Literal["sintel", "kitti", "chairs"]
