from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    index: int
    image: np.ndarray
