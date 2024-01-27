from dataclasses import dataclass
from typing import List

import cv2
import numpy
import numpy as np

from .edge_detection import EdgeDetector
from ..config import Config
from ..flow_utils.OpticalFlow import OpticalFlowProcessor
from ..flow_utils.warp import Warp


@dataclass
class Guides:
    edge: List[numpy.ndarray]
    flow_rev: List[numpy.ndarray]
    flow_fwd: List[numpy.ndarray]
    positional_rev: List[numpy.ndarray]
    positional_fwd: List[numpy.ndarray]


def create_guides(config: Config) -> Guides:
    edge_detector = EdgeDetector(method = config.edge_method)
    edge = [edge_detector.compute_edge(x) for i, x in config.images]

    optical_flow_processor = OpticalFlowProcessor(model_name = config.model_name, flow_method = config.flow_method)
    flow_rev = optical_flow_processor.compute_flow([x for i, x in config.images])
    flow_rev = [x for x in flow_rev]
    flow_fwd = [x * -1 for x in flow_rev]

    positional_rev = PositionalGuide([x for i, x in config.images], flow = flow_rev)
    positional_rev = positional_rev()

    positional_fwd = PositionalGuide([x for i, x in config.images], flow = flow_rev[::-1])
    positional_fwd = positional_fwd()
    positional_fwd = positional_fwd[::-1]

    return Guides(
        edge,
        flow_rev,
        flow_fwd,
        positional_rev,
        positional_fwd,
    )


class PositionalGuide:
    def __init__(self, images: List[numpy.ndarray], flow: List[numpy.ndarray]):
        self.coord_map = None
        self.coord_map_warped = None
        self.warp = Warp(images[0])  # Assuming Warp class has been modified to work with NumPy
        self.flow = flow
        self.images = images

    def __call__(self):
        self.g_pos = self._create_g_pos_from_flow(self.flow, self.images[0].shape[1::-1])
        return self.g_pos

    def _create_and_warp_coord_map(self, flow_up, original_size):
        if self.coord_map is None:
            h, w = self.warp.H, self.warp.W
            self.coord_map = np.zeros((h, w, 3), dtype = np.float32)
            self.coord_map[:, :, 0] = np.linspace(0, 1, w)
            self.coord_map[:, :, 1] = np.linspace(0, 1, h)[:, np.newaxis]
            self.coord_map_warped = self.coord_map.copy()
            return

        self.coord_map_warped = self.warp.run_warping(self.coord_map, flow_up)  # Assuming this returns a NumPy array

        if self.coord_map_warped is None:
            print("Warning: coord_map_warped is None!")
            return

        # Update the original coord_map with the newly warped version for the next iteration
        self.coord_map = self.coord_map_warped.copy()

    def _create_g_pos_from_flow(self, flow_np, original_size):
        g_pos_files = []
        for i in range(len(flow_np)):
            flow = flow_np[i - 1] if i != 0 else flow_np[i]
            self._create_and_warp_coord_map(flow, original_size)

            g_pos = cv2.resize(self.coord_map_warped, original_size)
            g_pos = np.clip(g_pos, 0, 1)
            g_pos = (g_pos * 255).astype(np.uint8)
            g_pos_files.append(g_pos)
        return g_pos_files
