import cv2
import numpy as np
import torch
from utils.guides.edge_detection import EdgeDetector
from utils.flow_utils.OpticalFlow import OpticalFlowProcessor
from utils.flow_utils.warp import Warp
#from utils import ebsynth, Preprocessor

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
    >>> factory = GuideFactory(imgsequence, edge_method="PAGE", flow_method="RAFT", model_name="sintel")
    >>> factory.create_all_guides()
    >>> custom_guides = some list of images as numpy arrays
    >>> factory.add_custom_guide("custom", custom_guides)
    """
    def __init__(self, imgsequence, edge_method="PAGE", flow_method="RAFT", model_name="sintel"):
        if not imgsequence:
            raise ValueError("Image sequence cannot be empty.")
        
        self.imgsequence = imgsequence
        self.edge_method = edge_method
        self.flow_method = flow_method
        self.model_name = model_name
        self.guides = {}
        
    def create_all_guides(self):
        edge_guide = EdgeGuide(self.imgsequence, method=self.edge_method)
        edge_guide = edge_guide()
        edge_guide = [edge for edge in edge_guide]
        flow_guide = FlowGuide(self.imgsequence, method=self.flow_method, model_name=self.model_name)
        flow_guide = flow_guide()
        flow_guide = [flow for flow in flow_guide]
        fwd_flow = FlowGuide(self.imgsequence[::-1], method=self.flow_method, model_name=self.model_name) # Reverse the image sequence
        fwd_flow = fwd_flow()   # Compute the flow, for some reason computing flow using imgseq backwards results in fwd_flow
        fwd_flow = [flow for flow in fwd_flow]
        positional_guide = PositionalGuide(self.imgsequence, flow=flow_guide)
        positional_guide = positional_guide()
        positional_fwd = PositionalGuide(self.imgsequence[::-1], flow=fwd_flow)
        positional_fwd = positional_fwd()
        positional_fwd = positional_fwd[::-1]
        fwd_flow = fwd_flow[::-1]
        for i in range(len(fwd_flow)):
            cv2.imwrite(f"C:/Users/tjerf/Desktop/Testing/src/Testvids/Output/positional_fwd{i}.jpg", positional_fwd[i])
            cv2.imwrite(f"C:/Users/tjerf/Desktop/Testing/src/Testvids/Output/positional_rev{i}.jpg", positional_guide[i])
            
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

    def __init__(self, imgseq, method="PAGE"):
        super().__init__()
        if method not in self.valid_methods:
            raise ValueError(f"Invalid method {method}. Valid methods are {self.valid_methods}")

        self.edge_detector = EdgeDetector(method)
        self.imgsequence = imgseq
        self.edge_maps = self._compute_edge()

    def __call__(self):
        return self.edge_maps

    def __iter__(self):
        for edge_map in self.edge_maps:
            yield edge_map

    def _compute_edge(self):
        edge_maps = []
        for img in self.imgsequence:
            edge_map = self._create(img)
            edge_maps.append(edge_map)
        return edge_maps

    def _create(self, img):
        return self.edge_detector.compute_edge(img) 
            
class FlowGuide(Guide):
    valid_methods = ["RAFT", "DeepFlow"]
    def __init__(self, imgseq, method = "RAFT", model_name = "sintel"):
        super().__init__()
        if method not in self.valid_methods:
            raise ValueError(f"Invalid method {method}. Valid methods are {self.valid_methods}")
        self.optical_flow_processor = OpticalFlowProcessor(model_name= model_name, flow_method = method)
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
        self.warp = Warp(imgseq[0])
        self.flow = flow
        self.imgseq = imgseq
        
    def __call__(self):
        return self._create()
    
    def _create_and_warp_coord_map(self, flow_up, original_size):
        flow_up = torch.from_numpy(flow_up).permute(2, 0, 1).float().unsqueeze(0)
        
        
        if self.coord_map is None:
            h, w = self.warp.H, self.warp.W
            self.coord_map = torch.zeros((1, 3, h, w))  # Added batch dimension
            
            # Debug: Print the shape of coord_map
            
            self.coord_map[0, 0] = torch.linspace(0, 1, w)
            self.coord_map[0, 1] = torch.linspace(0, 1, h)[:, np.newaxis]
            self.coord_map_warped = self.coord_map.clone()
            return 
        
        print("Warping coord map")
        self.coord_map_warped = self.warp._warp(self.coord_map, flow_up)
        
        if self.coord_map_warped is None:
            print("Warning: coord_map_warped is None!")
            return
        
        self.coord_map_warped = self.coord_map_warped.squeeze(0)
        
        # Update the original coord_map with the newly warped version for the next iteration
        self.coord_map = self.coord_map_warped.unsqueeze(0).clone()
    
    
    def _create_g_pos_from_flow(self, flow_np, original_size):
        original_size = original_size
        g_pos_files = []
        for i in range(len(flow_np)):
            print(f"iteration: {i}")
            flow = flow_np[i]
                    
            self._create_and_warp_coord_map(flow, original_size)
            
            if len(self.coord_map_warped.shape) == 4:
                g_pos = self.coord_map_warped.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            else:
                g_pos = self.coord_map_warped.permute(1, 2, 0).cpu().detach().numpy()
            
            g_pos = cv2.resize(g_pos, original_size)
            g_pos = np.clip(g_pos, 0, 1)
            g_pos = (g_pos * 255).astype(np.uint8)
            g_pos = cv2.cvtColor(g_pos, cv2.COLOR_BGR2RGB)
            g_pos_files.append(g_pos)
        return g_pos_files
    
    def _create(self):
        self.g_pos = self._create_g_pos_from_flow(self.flow, self.imgseq[0].shape[1::-1])
        return self.g_pos
        

    
    