import cv2
import numpy as np

from .edge_detection import EdgeDetector
from ..config import Config, Guides
from ..flow_utils.OpticalFlow import OpticalFlowProcessor
from ..flow_utils.warp import Warp


def create_guides(config: Config) -> Guides:
    edge_detector = EdgeDetector(config.edge_method)
    edge_guide = [edge_detector.compute_edge(x) for i, x in config.images]

    flow_guide = FlowGuide(config.imgsequence, method = config.flow_method, model_name = config.model_name)
    flow_guide = flow_guide()
    flow_guide = [flow for flow in flow_guide]
    # fwd_flow = FlowGuide(self.imgsequence[::-1], method=self.flow_method, model_name=self.model_name) # Reverse the image sequence
    # fwd_flow = fwd_flow()   # Compute the flow, for some reason computing flow using imgseq backwards results in fwd_flow
    # fwd_flow = [flow for flow in fwd_flow]
    positional_guide = PositionalGuide(config.imgsequence, flow = flow_guide)
    positional_guide = positional_guide()
    positional_fwd = PositionalGuide(config.imgsequence, flow = flow_guide[::-1])
    positional_fwd = positional_fwd()
    positional_fwd = positional_fwd[::-1]
    fwd_flow = [flow * -1 for flow in flow_guide]

    return Guides(
        edge_guide,
        flow_guide,
        fwd_flow,
        positional_guide,
        positional_fwd,
    )


class FlowGuide:
    valid_methods = ["RAFT", "DeepFlow"]

    def __init__(self, imgseq, method = "RAFT", model_name = "sintel"):
        if method not in self.valid_methods:
            raise ValueError(f"Invalid method {method}. Valid methods are {self.valid_methods}")
        self.optical_flow_processor = OpticalFlowProcessor(model_name = model_name, flow_method = method)
        self.imgsequence = imgseq
        self.optical_flow = None

    def __call__(self):
        return self._create()

    def _create(self):
        self.optical_flow = self.optical_flow_processor.compute_flow(self.imgsequence)
        return self.optical_flow


class PositionalGuide:
    def __init__(self, imgseq, flow):
        self.coord_map = None
        self.coord_map_warped = None
        self.warp = Warp(imgseq[0])  # Assuming Warp class has been modified to work with NumPy
        self.flow = flow
        self.imgseq = imgseq

    def __call__(self):
        return self._create()

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

    def _create(self):
        self.g_pos = self._create_g_pos_from_flow(self.flow, self.imgseq[0].shape[1::-1])
        return self.g_pos
