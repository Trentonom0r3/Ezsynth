import cv2
import numpy as np

from .edge_detection import EdgeDetector
from ..flow_utils.OpticalFlow import OpticalFlowProcessor
from ..flow_utils.warp import Warp


# from utils import ebsynth, Preprocessor

class GuideFactory():
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
    VALID_FLOW_METHODS = ["RAFT", "DeepFlow"]
    VALID_MODEL_NAMES = ["sintel", "kitti", "chairs"]

    def __init__(self, imgsequence, imgseq, edge_method = "PAGE", flow_method = "RAFT", model_name = "sintel"):
        if not imgsequence:
            raise ValueError("Image sequence cannot be empty.")

        self.imgsequence = imgsequence
        self.edge_method = edge_method if edge_method in self.VALID_EDGE_METHODS else "PAGE"
        self.flow_method = flow_method if flow_method in self.VALID_FLOW_METHODS else "RAFT"
        self.model_name = model_name if model_name in self.VALID_MODEL_NAMES else "sintel"
        self.guides = {}
        self.imgs = imgseq

    def create_all_guides(self):

        edge_guide = EdgeGuide(self.imgs, method = self.edge_method)
        edge_guide = edge_guide()
        # edge_guide = [edge for edge in edge_guide]
        flow_guide = FlowGuide(self.imgsequence, method = self.flow_method, model_name = self.model_name)
        flow_guide = flow_guide()
        flow_guide = [flow for flow in flow_guide]
        # fwd_flow = FlowGuide(self.imgsequence[::-1], method=self.flow_method, model_name=self.model_name) # Reverse the image sequence
        # fwd_flow = fwd_flow()   # Compute the flow, for some reason computing flow using imgseq backwards results in fwd_flow
        # fwd_flow = [flow for flow in fwd_flow]
        positional_guide = PositionalGuide(self.imgsequence, flow = flow_guide)
        positional_guide = positional_guide()
        positional_fwd = PositionalGuide(self.imgsequence, flow = flow_guide[::-1])
        positional_fwd = positional_fwd()
        positional_fwd = positional_fwd[::-1]
        fwd_flow = [flow * -1 for flow in flow_guide]

        self.guides = {
            "edge": edge_guide,
            "flow_rev": flow_guide,
            "flow_fwd": fwd_flow,
            "positional_rev": positional_guide,
            "positional_fwd": positional_fwd,
        }

        return self.guides

    def add_custom_guide(self, name, custom_guides):
        if len(custom_guides) != len(self.imgsequence):
            raise ValueError("The length of the custom guide must match the length of the image sequence.")

        self.guides[name] = custom_guides

    def __call__(self):
        return self.create_all_guides()


class Guide():
    def __init__(self):
        pass


class EdgeGuide(Guide):
    valid_methods = ["PAGE", "PST", "Classic"]

    def __init__(self, imgseq, method = "PAGE"):
        super().__init__()
        if method not in self.valid_methods:
            raise ValueError(f"Invalid method {method}. Valid methods are {self.valid_methods}")

        self.edge_detector = EdgeDetector(method)
        self.imgseq = imgseq

        self._edge_maps = None  # Store edge_maps here when computed

    def __call__(self):
        if self._edge_maps is None:
            self._compute_edge()
        return self._edge_maps

    def __iter__(self):
        if self._edge_maps is None:
            self._compute_edge()
        for edge_map in self._edge_maps:
            yield edge_map

    def _compute_edge(self):
        # Uncomment the following line to potentially parallelize this computation
        # with concurrent.futures.ThreadPoolExecutor() as executor:

        self._edge_maps = [self._create(img) for img in self.imgseq]

    def _create(self, img):

        return self.edge_detector.compute_edge(img)


class FlowGuide(Guide):
    valid_methods = ["RAFT", "DeepFlow"]

    def __init__(self, imgseq, method = "RAFT", model_name = "sintel"):
        super().__init__()
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


class PositionalGuide(Guide):
    def __init__(self, imgseq, flow):
        super().__init__()
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
