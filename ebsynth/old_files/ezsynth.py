import os
import sys
from ctypes import *
from pathlib import Path
import cv2
import numpy as np

class ebsynth:
    # Constants as Class Attributes
    EBSYNTH_BACKEND_CPU = 0x0001
    EBSYNTH_BACKEND_CUDA = 0x0002
    EBSYNTH_BACKEND_AUTO = 0x0000
    EBSYNTH_MAX_STYLE_CHANNELS = 8
    EBSYNTH_MAX_GUIDE_CHANNELS = 24
    EBSYNTH_VOTEMODE_PLAIN = 0x0001
    EBSYNTH_VOTEMODE_WEIGHTED = 0x0002

    def __init__(self, style=None, guides=[], weight=None, uniformity=3500.0, 
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
            self.weight = weight if weight else [1.0 for _ in range(len(guides))]
            self.uniformity = uniformity
            self.patchsize = patchsize
            self.pyramidlevels = pyramidlevels
            self.searchvoteiters = searchvoteiters
            self.patchmatchiters = patchmatchiters
            self.extrapass3x3 = extrapass3x3
            self.nnf_buffer = nnf_buffer
            self.stop_threshold = 1
            # Define backend constants
            self.backends = {
                'cpu': self.EBSYNTH_BACKEND_CPU,
                'cuda': self.EBSYNTH_BACKEND_CUDA,
                'auto': self.EBSYNTH_BACKEND_AUTO
            }
            self.backend = self.backends[backend]

            self.buffer_pool = {}
            self.libebsynth = None
            self.cached_buffer = {}

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

    def _get_buffer(self, shape):
        buffer = self.buffer_pool.get(shape)
        if buffer is None:
            buffer = create_string_buffer(shape[0] * shape[1] * shape[2])
            self.buffer_pool[shape] = buffer
        return buffer

    def _normalize_img_shape(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
        elif isinstance(img, np.ndarray):
            pass
            
        img_len = len(img.shape)
        if img_len == 2:
            sh, sw = img.shape
            sc = 0
        elif img_len == 3:
            sh, sw, sc = img.shape

        if sc == 0:
            sc = 1
            img = img[..., np.newaxis]
        return img

    def run (self):
        if self.patchsize < 3:
            raise ValueError ("patch_size is too small")
        if self.patchsize % 2 == 0:
            raise ValueError ("patch_size must be an odd number")
        if len(self.guides) == 0:
            raise ValueError ("at least one guide must be specified")

  
        if self.libebsynth is None:
            if sys.platform[0:3] == 'win':
                libebsynth_path = 'ebsynth.dll'
                self.libebsynth = CDLL(libebsynth_path)
            else:
                #todo: implement for linux
                pass

            if self.libebsynth is not None:
                self.libebsynth.ebsynthRun.argtypes = ( \
                    c_int,
                    c_int,
                    c_int,
                    c_int,
                    c_int,
                    c_void_p,
                    c_void_p,
                    c_int,
                    c_int,
                    c_void_p,
                    c_void_p,
                    POINTER(c_float),
                    POINTER(c_float),
                    c_float,
                    c_int,
                    c_int,
                    c_int,
                    POINTER(c_int),
                    POINTER(c_int),
                    POINTER(c_int),
                    c_int,
                    c_void_p,
                    c_void_p
                    )

        if self.libebsynth is None:
            return img_style

        img_style = self._normalize_img_shape (self.style)
        sh, sw, sc = img_style.shape
        t_h, t_w, t_c = 0,0,0

        if sc > self.EBSYNTH_MAX_STYLE_CHANNELS:
            raise ValueError (f"error: too many style channels {sc}, maximum number is {self.EBSYNTH_MAX_STYLE_CHANNELS}")

        guides_source = []
        guides_target = []
        guides_weights = []

        for i in range(len(self.guides)):
            source_guide, target_guide, guide_weight = self.guides[i]
            source_guide = self._normalize_img_shape(source_guide)
            target_guide = self._normalize_img_shape(target_guide)
            s_h, s_w, s_c = source_guide.shape
            nt_h, nt_w, nt_c = target_guide.shape

            if s_h != sh or s_w != sw:
                raise ValueError("guide source and style resolution must match style resolution.")

            if t_c == 0:
                t_h, t_w, t_c = nt_h, nt_w, nt_c
            elif nt_h != t_h or nt_w != t_w:
                raise ValueError("guides target resolutions must be equal")

            if s_c != nt_c:
                raise ValueError("guide source and target channels must match exactly.")

            guides_source.append(source_guide)
            guides_target.append(target_guide)

            guide_weight_scaled = guide_weight / s_c  # Compute the division once
            guides_weights += [guide_weight_scaled] * s_c  # Use the stored value

        guides_source = np.concatenate(guides_source, axis=-1)
        guides_target = np.concatenate(guides_target, axis=-1)
        guides_weights = (c_float*len(guides_weights))(*guides_weights)

        styleWeight = 1.0
        style_weights = [styleWeight / sc for i in range(sc)]
        style_weights = (c_float*sc)(*style_weights)



        maxPyramidLevels = 0
        for level in range(32,-1,-1):
            if min( min(sh, t_h)*pow(2.0, -level), \
                    min(sw, t_w)*pow(2.0, -level) ) >= (2*self.patchsize+1):
                maxPyramidLevels = level+1
                break

        if self.pyramidlevels == -1:
            self.pyramidlevels = maxPyramidLevels
        num_pyramid_levels = min(self.pyramidlevels, maxPyramidLevels)

        num_search_vote_iters_per_level = (c_int*num_pyramid_levels) ( *[self.searchvoteiters]*num_pyramid_levels )
        num_patch_match_iters_per_level = (c_int*num_pyramid_levels) ( *[self.patchmatchiters]*num_pyramid_levels )
        stop_threshold_per_level = (c_int*num_pyramid_levels) ( *[self.stop_threshold]*num_pyramid_levels )
        
        if self.nnf_buffer is not None:
            nnf_buffer = create_string_buffer(t_h * t_w * 2 * sizeof(c_int))
        else:
            nnf_buffer = None
        
        buffer = self.cached_buffer.get ( (t_h,t_w,sc), None )
        if buffer is None:
            buffer_shape = (t_h, t_w, sc)
            buffer = self._get_buffer(buffer_shape)
            self.cached_buffer[(t_h,t_w,sc)] = buffer

        self.libebsynth.ebsynthRun (self.backend,    #backend
                            sc,                      #numStyleChannels
                            guides_source.shape[-1], #numGuideChannels
                            sw,                      #sourceWidth
                            sh,                      #sourceHeight
                            img_style.tobytes(),     #sourceStyleData (width * height * numStyleChannels) bytes, scan-line order
                            guides_source.tobytes(), #sourceGuideData (width * height * numGuideChannels) bytes, scan-line order
                            t_w,                     #targetWidth
                            t_h,                     #targetHeight
                            guides_target.tobytes(), #targetGuideData (width * height * numGuideChannels) bytes, scan-line order
                            None,                    #targetModulationData (width * height * numGuideChannels) bytes, scan-line order; pass NULL to switch off the modulation
                            style_weights,           #styleWeights (numStyleChannels) floats
                            guides_weights,          #guideWeights (numGuideChannels) floats
                            self.uniformity,       #uniformityWeight reasonable values are between 500-15000, 3500 is a good default
                            self.patchsize,              #patchSize odd sizes only, use 5 for 5x5 patch, 7 for 7x7, etc.
                            self.EBSYNTH_VOTEMODE_WEIGHTED,  #voteMode use VOTEMODE_WEIGHTED for sharper result
                            num_pyramid_levels,      #numPyramidLevels

                            num_search_vote_iters_per_level, #numSearchVoteItersPerLevel how many search/vote iters to perform at each level (array of ints, coarse first, fine last)
                            num_patch_match_iters_per_level, #numPatchMatchItersPerLevel how many Patch-Match iters to perform at each level (array of ints, coarse first, fine last)
                            stop_threshold_per_level, #stopThresholdPerLevel stop improving pixel when its change since last iteration falls under this threshold
                            1 if self.extrapass3x3 else 0, #extraPass3x3 perform additional polishing pass with 3x3 patches at the finest level, use 0 to disable
                            nnf_buffer,                     #outputNnfData (width * height * 2) ints, scan-line order; pass NULL to ignore
                            buffer                    #outputImageData  (width * height * numStyleChannels) bytes, scan-line order
                            )
        
        if nnf_buffer is not None:
            return np.frombuffer(buffer, dtype=np.uint8).reshape ( (t_h,t_w,sc) ).copy(), np.frombuffer(nnf_buffer, dtype=np.int32).reshape((t_h, t_w, 2))
    
        else:
            return np.frombuffer(buffer, dtype=np.uint8).reshape ( (t_h,t_w,sc) ).copy()
