from typing import List

import numpy


class Config:
    styles: List[tuple[int, numpy.ndarray]]
    images: List[tuple[int, numpy.ndarray]]
    edge_method: str
    flow_method: str
    model_name: str
