import os
import re
import cv2
import numpy as np
import torch
from tqdm import tqdm
from _ebsynth import run, color_transfer, EBSYNTH_BACKEND_CPU, EBSYNTH_BACKEND_CUDA, EBSYNTH_BACKEND_AUTO
from utils.edge_detection import EdgeDetector
from utils.optical_flow import OpticalFlowProcessor

class ebsynth:
    """
    EBSynth class provides a wrapper around the ebsynth style transfer method.

    Usage:
        ebsynth = ebsynth.ebsynth(style='style.png', guides=[('source1.png', 'target1.png'), 1.0])
        result_img = ebsynth.run()
    """
    
    def __init__(self, style = None, guides=[], weight=None, uniformity=3500.0, 
             patchsize=5, pyramidlevels=6, searchvoteiters=12, 
             patchmatchiters=6, extrapass3x3=True, backend='cuda', nnf_buffer=None):

        """
        Initialize the EBSynth wrapper.      
        :param style: path to the style image, or a numpy array.
        :param guides: list of tuples containing source and target guide images, as file paths or as numpy arrays.
        :param weight: weights for each guide pair. Defaults to 1.0 for each pair.
        :param uniformity: uniformity weight for the style transfer. Defaults to 3500.0.
        :param patchsize: size of the patches. Must be an odd number. Defaults to 5. [5x5 patches]
        :param pyramidlevels: number of pyramid levels. Larger Values useful for things like color transfer. Defaults to 6.
        :param searchvoteiters: number of search/vote iterations. Defaults to 12.
        :param patchmatchiters: number of Patch-Match iterations. Defaults to 6.
        :param extrapass3x3: whether to perform an extra pass with 3x3 patches. Defaults to False.
        :param backend: backend to use ('cpu', 'cuda', or 'auto'). Defaults to 'auto'.
        """
          # Handling the style image
        if isinstance(style, (np.ndarray)):
            self.style = style
        elif isinstance(style, (str)):
            self.style = cv2.imread(style)
        elif style is None:
            print("[INFO] No Style Image Provided. Remember to add a style image to the run() method.")
        else:
            print(type(style))
            raise ValueError("style should be either a file path or a numpy array.")

        # Handling the guide images
        self.guides = []
        for guide in guides:
            if not isinstance(guide[0], (str, np.ndarray)):
                raise ValueError("source should be either a file path or a numpy array.")
            if not isinstance(guide[1], (str, np.ndarray)):
                raise ValueError("target should be either a file path or a numpy array.")
            if len(guide) == 3 and isinstance(guide[2], (float, int)):
                self.guides.append(guide)
            else:
                self.guides.append((guide[0], guide[1], 1.0))

        if weight:
            if len(weight) != len(self.guides):
                raise ValueError("The number of weights should match the number of guide pairs.")
            for i, guide in enumerate(self.guides):
                self.guides[i] = (guide[0], guide[1], weight[i])
        else:
            for i, guide in enumerate(self.guides):
                if len(guide) < 3:
                    self.guides[i] = (guide[0], guide[1], 1.0)
        
        # Store the arguments
        self.style = style
        self.guides = guides
        self.weight = weight if weight else [1.0 for _ in range(len(guides))]
        self.uniformity = uniformity
        self.patchsize = patchsize
        self.pyramidlevels = pyramidlevels
        self.searchvoteiters = searchvoteiters
        self.patchmatchiters = patchmatchiters
        self.extrapass3x3 = extrapass3x3
        self.nnf_buffer = nnf_buffer
        # Define backend constants
        self.backends = {
            'cpu': EBSYNTH_BACKEND_CPU,
            'cuda': EBSYNTH_BACKEND_CUDA,
            'auto': EBSYNTH_BACKEND_AUTO
        }
        self.backend = self.backends[backend]

    def add_guide(self, source, target, weight=None):
        """
        Add a new guide pair.
        
        :param source: Path to the source guide image or a numpy array.
        :param target: Path to the target guide image or a numpy array.
        :param weight: Weight for the guide pair. Defaults to 1.0.
        """
        if not isinstance(source, (str, np.ndarray)):
            raise ValueError("source should be either a file path or a numpy array.")
        if not isinstance(target, (str, np.ndarray)):
            raise ValueError("target should be either a file path or a numpy array.")
        weight = weight if weight is not None else 1.0
        self.guides.append((source, target, weight))


    def run(self):
        """
        Run the style transfer and return the result image.
        
        :return: styled image as a numpy array.
        """
        if isinstance(self.style, np.ndarray):
            img_style = self.style
        else:
            img_style = cv2.imread(self.style)

        # Prepare the guides
        guides_processed = []
        for idx, (source, target, weight) in enumerate(self.guides):
            if isinstance(source, np.ndarray):
                source_img = source
            else:
                source_img = cv2.imread(source)
            if isinstance(target, np.ndarray):
                target_img = target
            else:
                target_img = cv2.imread(target)
            guides_processed.append((source_img, target_img, weight or self.weight[idx]))

        # Call the run function with the provided arguments
        result = run(img_style, guides_processed, 
                    patch_size=self.patchsize,
                    num_pyramid_levels=self.pyramidlevels,
                    num_search_vote_iters=self.searchvoteiters,
                    num_patch_match_iters=self.patchmatchiters,
                    uniformity_weight=self.uniformity,
                    extraPass3x3=self.extrapass3x3,
                    backend=self.backend, nnf_buffer=self.nnf_buffer
                    )

        return result
    
    def color_transfer(self, source, target):
        guides = [{cv2.cvtcolor(source, cv2.COLOR_BGR2LAB), 
                   cv2.cvtcolor(target, cv2.COLOR_BGR2LAB), 1}]
        h,w,c = source.shape
        result = []
        for i in range(c):
            result += [        
                        run(source[...,i:i+1] , guides=guides, 
                                    patch_size=11, 
                                    num_pyramid_levels=40, 
                                    num_search_vote_iters = 6,
                                    num_patch_match_iters = 4,
                                    stop_threshold = 5,
                                    uniformity_weight=500.0,
                                    extraPass3x3=True,
                                    )
                                    
                    ]
        return np.concatenate( result, axis=-1 )

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////------------------HELPER_CLASSES------------------//////////////////////////////////////////

class Sequence:
    """
    Helper class to store sequence information.
    
    :param begFrame: Index of the first frame in the sequence.
    :param endFrame: Index of the last frame in the sequence.
    :param keyframeIdx: Index of the keyframe in the sequence.
    :param style_image: Style image for the sequence.
    
    :return: Sequence object.
    
    """
    def __init__(self, begFrame, endFrame, style_start = None, style_end = None):
        self.begFrame = begFrame
        self.endFrame = endFrame
        self.style_start = style_start if style_start else None
        self.style_end = style_end if style_end else None
        if self.style_start and self.style_end == None:
            raise ValueError("At least one style attribute should be provided.")
        

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////   
