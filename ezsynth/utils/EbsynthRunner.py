import os.path
import sys
import threading
from ctypes import *

import numpy as np


class EbsynthRunner:
    EBSYNTH_BACKEND_CPU = 0x0001
    EBSYNTH_BACKEND_CUDA = 0x0002
    EBSYNTH_BACKEND_AUTO = 0x0000
    EBSYNTH_MAX_STYLE_CHANNELS = 8
    EBSYNTH_MAX_GUIDE_CHANNELS = 24
    EBSYNTH_VOTEMODE_PLAIN = 0x0001  # weight = 1
    EBSYNTH_VOTEMODE_WEIGHTED = 0x0002  # weight = 1/(1+error)

    def __init__(self):
        self.lib = None
        self.cached_buffer = {}
        self.cached_err_buffer = {}
        self.lock = threading.Lock()
        self.cache_lock = threading.Lock()
        self.normalize_lock = threading.Lock()

    def init_lib(self):
        with self.lock:
            if self.lib is None:
                if sys.platform[0:3] == 'win':
                    self.lib = CDLL(os.path.join(__file__, "..", 'ebsynth.dll'))
                elif sys.platform == 'darwin':
                    self.lib = CDLL(os.path.join(__file__, "..", 'ebsynth.so'))
                elif sys.platform[0:5] == 'linux':
                    self.lib = CDLL(os.path.join(__file__, "..", 'ebsynth.so'))

                if self.lib is None:
                    raise RuntimeError("Unsupported platform.")

                self.lib.ebsynthRun.argtypes = (
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
                    c_void_p,
                    c_void_p
                )

    def get_or_create_buffer(self, key):
        with self.cache_lock:
            a = self.cached_buffer.get(key, None)
            if a is None:
                a = create_string_buffer(key[0] * key[1] * key[2])
                self.cached_buffer[key] = a
            return a

    def get_or_create_err_buffer(self, key):
        with self.cache_lock:
            a = self.cached_err_buffer.get(key, None)
            if a is None:
                a = (c_float * (key[0] * key[1]))()
                self.cached_err_buffer[key] = a
            return a

    def _normalize_img_shape(self, img):
        with self.normalize_lock:
            if len(img.shape) == 2:
                sc = 0
            elif len(img.shape) == 3:
                sh, sw, sc = img.shape

            if sc == 0:
                img = img[..., np.newaxis]

            return img

    def run(
            self,
            img_style,
            guides,
            patch_size = 5,
            num_pyramid_levels = -1,
            num_search_vote_iters = 6,
            num_patch_match_iters = 4,
            stop_threshold = 5,
            uniformity_weight = 3500.0,
            extra_pass3x3 = False,
    ):

        # Validation checks
        if patch_size < 3:
            raise ValueError("patch_size is too small")
        if patch_size % 2 == 0:
            raise ValueError("patch_size must be an odd number")
        if len(guides) == 0:
            raise ValueError("at least one guide must be specified")

        # Initialize libebsynth if not already done
        self.init_lib()

        img_style = self._normalize_img_shape(img_style)
        sh, sw, sc = img_style.shape
        t_h, t_w, t_c = 0, 0, 0

        if sc > self.EBSYNTH_MAX_STYLE_CHANNELS:
            raise ValueError(
                f"error: too many style channels {sc}, maximum number is {self.EBSYNTH_MAX_STYLE_CHANNELS}")

        guides_source = []
        guides_target = []
        guides_weights = []

        for i in range(len(guides)):
            source_guide, target_guide, guide_weight = guides[i]
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

            guides_weights += [guide_weight / s_c] * s_c

        guides_source = np.concatenate(guides_source, axis = -1)
        guides_target = np.concatenate(guides_target, axis = -1)
        guides_weights = (c_float * len(guides_weights))(*guides_weights)

        style_weights = [1.0 / sc for i in range(sc)]
        style_weights = (c_float * sc)(*style_weights)

        maxPyramidLevels = 0
        for level in range(32, -1, -1):
            if min(min(sh, t_h) * pow(2.0, -level), min(sw, t_w) * pow(2.0, -level)) >= (2 * patch_size + 1):
                maxPyramidLevels = level + 1
                break

        if num_pyramid_levels == -1:
            num_pyramid_levels = maxPyramidLevels
        num_pyramid_levels = min(num_pyramid_levels, maxPyramidLevels)

        num_search_vote_iters_per_level = (c_int * num_pyramid_levels)(*[num_search_vote_iters] * num_pyramid_levels)
        num_patch_match_iters_per_level = (c_int * num_pyramid_levels)(*[num_patch_match_iters] * num_pyramid_levels)
        stop_threshold_per_level = (c_int * num_pyramid_levels)(*[stop_threshold] * num_pyramid_levels)

        # Get or create buffers
        buffer = self.get_or_create_buffer((t_h, t_w, sc))
        errbuffer = self.get_or_create_err_buffer((t_h, t_w))

        with self.lock:
            self.lib.ebsynthRun(
                self.EBSYNTH_BACKEND_AUTO,
                sc,  # numStyleChannels
                guides_source.shape[-1],  # numGuideChannels
                sw,  # sourceWidth
                sh,  # sourceHeight
                img_style.tobytes(),
                # sourceStyleData (width * height * numStyleChannels) bytes, scan-line order
                guides_source.tobytes(),
                # sourceGuideData (width * height * numGuideChannels) bytes, scan-line order
                t_w,  # targetWidth
                t_h,  # targetHeight
                guides_target.tobytes(),
                # targetGuideData (width * height * numGuideChannels) bytes, scan-line order
                None,
                # targetModulationData (width * height * numGuideChannels) bytes, scan-line order; pass NULL to switch off the modulation
                style_weights,  # styleWeights (numStyleChannels) floats
                guides_weights,  # guideWeights (numGuideChannels) floats
                uniformity_weight,
                # uniformityWeight reasonable values are between 500-15000, 3500 is a good default
                patch_size,  # patchSize odd sizes only, use 5 for 5x5 patch, 7 for 7x7, etc.
                self.EBSYNTH_VOTEMODE_WEIGHTED,
                # voteMode use VOTEMODE_WEIGHTED for sharper result
                num_pyramid_levels,  # numPyramidLevels
                num_search_vote_iters_per_level,
                # numSearchVoteItersPerLevel how many search/vote iters to perform at each level (array of ints, coarse first, fine last)
                num_patch_match_iters_per_level,
                # numPatchMatchItersPerLevel how many Patch-Match iters to perform at each level (array of ints, coarse first, fine last)
                stop_threshold_per_level,
                # stopThresholdPerLevel stop improving pixel when its change since last iteration falls under this threshold
                1 if extra_pass3x3 else 0,
                # extraPass3x3 perform additional polishing pass with 3x3 patches at the finest level, use 0 to disable
                None,
                # outputNnfData (width * height * 2) ints, scan-line order; pass NULL to ignore
                buffer,
                # outputImageData  (width * height * numStyleChannels) bytes, scan-line order
                errbuffer,
                # outputErrorData (width * height) floats, scan-line order; pass NULL to ignore
            )

        img = np.frombuffer(buffer, dtype = np.uint8).reshape((t_h, t_w, sc)).copy()
        err = np.frombuffer(errbuffer, dtype = np.float32).reshape((t_h, t_w)).copy()

        return img, err
