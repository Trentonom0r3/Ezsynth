import sys
import time

import cv2
import numpy as np
import tqdm

from ..flow_utils.OpticalFlow import OpticalFlowProcessor
from ..flow_utils.warp import Warp

# import torch
from .edge_detection import EdgeDetector

# from utils import ebsynth, Preprocessor


class GuideFactory:
    """
    Factory class for creating and managing different types of guides.

    The factory class provides methods to create different types of guides
    like edge guides, flow guides, and positional guides. It also allows
    the addition of custom guides.

    Parameters
    ----------
    imgsequence : list
        The sequence of images for which the guides will be created.
    edge_method : str, optional
        The method for edge detection, default is "PAGE".
    flow_method : str, optional
        The method for optical flow computation, default is "RAFT".
    model_name : str, optional
        The model name for optical flow, default is "sintel".

    Attributes
    ----------
    imgsequence : list
        The sequence of images for which the guides are created.
    edge_method : str
        The method used for edge detection.
    flow_method : str
        The method used for optical flow computation.
    model_name : str
        The name of the model used for optical flow.
    guides : dict
        Dictionary to store the created guide objects.

    Methods
    -------
    create_all_guides()
        Create all default guides.
    add_custom_guide(name, custom_guide)
        Add a custom guide to the factory's collection of guides.

    Usage
    -----
        factory = GuideFactory(imgsequence, edge_method="PAGE", flow_method="RAFT", model_name="sintel")
        factory.create_all_guides()
        custom_guides = some list of images as numpy arrays
        factory.add_custom_guide("custom", custom_guides)
    """

    VALID_EDGE_METHODS = ["PAGE", "PST", "Classic"]
    VALID_FLOW_METHODS = [
        "RAFT",
        #   "DeepFlow" # There is no support for deepflow in the original repo
    ]
    VALID_MODEL_NAMES = ["sintel", "kitti", "chairs"]

    DEFAULT_EDGE_METHOD = "PAGE"
    DEFAULT_FLOW_METHOD = "RAFT"
    DEFAULT_MODEL_NAME = "sintel"

    @classmethod
    def validate_edge_flow_model(
        cls, edge_method: str, flow_method: str, model_name: str
    ):
        edge_method = (
            edge_method
            if edge_method in cls.VALID_EDGE_METHODS
            else cls.DEFAULT_EDGE_METHOD
        )
        flow_method = (
            flow_method
            if flow_method in cls.VALID_FLOW_METHODS
            else cls.DEFAULT_FLOW_METHOD
        )
        model_name = (
            model_name
            if model_name in cls.VALID_MODEL_NAMES
            else cls.DEFAULT_MODEL_NAME
        )
        return edge_method, flow_method, model_name

    def __init__(
        self,
        img_frs_seq: list[np.ndarray],
        img_file_paths: list[str],
        edge_method="PAGE",
        flow_method="RAFT",
        model_name="sintel",
    ):
        if not img_frs_seq:
            raise ValueError("Image sequence cannot be empty.")

        self.img_frs_seq = img_frs_seq
        self.edge_method, self.flow_method, self.model_name = (
            self.validate_edge_flow_model(edge_method, flow_method, model_name)
        )

        self.guides = {}
        self.img_file_paths = img_file_paths

    @staticmethod
    def create_edge_guide(img_file_paths: list[str], edge_method: str):
        pass

    def create_all_guides(self):
        st = time.time()
        edge_guide = EdgeGuide(self.img_file_paths, method=self.edge_method)
        edge_guide = edge_guide.run(self.img_frs_seq)
        print(f"Edge guide took {time.time() - st:.4f} s")
        st = time.time()
        flow_guide = FlowGuide(
            self.img_frs_seq, method=self.flow_method, model_name=self.model_name
        )
        flow_guide = [flow for flow in flow_guide._create()]
        print(f"Flow guide took {time.time() - st:.4f} s")
        fwd_flow = FlowGuide(
            list(reversed(self.img_frs_seq)), method=self.flow_method, model_name=self.model_name
        )  # Reverse the image sequence
        fwd_flow = fwd_flow._create()  # Compute the flow, for some reason computing flow using imgseq backwards results in fwd_flow
        st = time.time()
        positional_guide = PositionalGuide(self.img_frs_seq, flow=flow_guide)
        positional_guide = positional_guide()
        positional_fwd = PositionalGuide(self.img_frs_seq, flow=flow_guide[::-1])
        positional_fwd = positional_fwd()
        positional_fwd = positional_fwd[::-1]
        fwd_flow = [flow * -1 for flow in flow_guide]
        print(f"Pos guide took {time.time() - st:.4f} s")

        self.guides = {
            "edge": edge_guide,
            "flow_rev": flow_guide,
            "flow_fwd": fwd_flow,
            "positional_rev": positional_guide,
            "positional_fwd": positional_fwd,
        }

        return self.guides

    def add_custom_guide(self, name, custom_guides):
        if len(custom_guides) != len(self.img_frs_seq):
            raise ValueError(
                "The length of the custom guide must match the length of the image sequence."
            )

        self.guides[name] = custom_guides


class EdgeGuide:
    def __init__(self, img_file_paths, method="PAGE"):
        self.edge_detector = EdgeDetector(method)
        self.img_file_paths = img_file_paths

        self._edge_maps = None  # Store edge_maps here when computed

    def run(self, img_frs_seq: np.ndarray):
        edge_maps = []
        for img_fr in tqdm.tqdm(img_frs_seq, desc="Calculating edge maps"):
            edge_maps.append(self.edge_detector.compute_edge(img_fr))
        self._edge_maps = edge_maps
        return self._edge_maps


class FlowGuide:
    def __init__(self, img_frs_seq, method="RAFT", model_name="sintel"):
        self.optical_flow_processor = OpticalFlowProcessor(
            model_name=model_name, flow_method=method
        )
        self.img_frs_seq = img_frs_seq
        self.optical_flow = None

    def _create(self):
        self.optical_flow = self.optical_flow_processor.compute_flow(self.img_frs_seq)
        return self.optical_flow


class PositionalGuide:
    def __init__(self, imgseq, flow):
        self.coord_map = None
        self.coord_map_warped = None
        self.warp = Warp(
            imgseq[0]
        )  # Assuming Warp class has been modified to work with NumPy
        self.flow = flow
        self.imgseq = imgseq

    def __call__(self):
        return self._create()

    def _create_and_warp_coord_map(self, flow_up, original_size):
        if self.coord_map is None:
            h, w = self.warp.H, self.warp.W
            self.coord_map = np.zeros((h, w, 3), dtype=np.float32)
            self.coord_map[:, :, 0] = np.linspace(0, 1, w)
            self.coord_map[:, :, 1] = np.linspace(0, 1, h)[:, np.newaxis]
            self.coord_map_warped = self.coord_map.copy()
            return

        self.coord_map_warped = self.warp.run_warping(
            self.coord_map, flow_up
        )  # Assuming this returns a NumPy array

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
        self.g_pos = self._create_g_pos_from_flow(
            self.flow, self.imgseq[0].shape[1::-1]
        )
        return self.g_pos
