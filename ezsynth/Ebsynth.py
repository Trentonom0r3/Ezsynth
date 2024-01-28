import os
import sys
import threading
from ctypes import *
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np


@dataclass
class Config:
    """
    Ebsynth config.
    :param style_image: Path to the image, or a numpy array.
    :param guides: List of tuples containing: source image, target image, weight.
    :param uniformity: Uniformity weight for the style transfer. Defaults to 3500.
    :param patch_size: Size of the patches. Must be an odd number. Defaults to 5 (5x5 patches).
    :param num_pyramid_levels: Number of pyramid levels. Larger Values useful for things like color transfer. Defaults to 6.
    :param num_search_vote_iters: Number of search/vote iterations. Defaults to 12.
    :param num_patch_match_iters: Number of Patch-Match iterations. Defaults to 6.
    :param stop_threshold: Stop threshold. Defaults to 5.
    :param extra_pass3x3: Whether to perform an extra pass with 3x3 patches. Defaults to False.
    """

    style_image: Union[str, np.ndarray]
    guides: List[Tuple[Union[str, np.ndarray], Union[str, np.ndarray], Union[int, float, None]]]
    uniformity: float = 3500.0
    patch_size: int = 5
    num_pyramid_levels: int = 6
    num_search_vote_iters: int = 12
    num_patch_match_iters: int = 6
    stop_threshold: int = 5
    extra_pass3x3: bool = False


class Ebsynth:
    BACKEND_CPU = 0x0001
    BACKEND_CUDA = 0x0002
    BACKEND_AUTO = 0x0000
    MAX_STYLE_CHANNELS = 8
    MAX_GUIDE_CHANNELS = 24
    VOTEMODE_PLAIN = 0x0001  # weight = 1
    VOTEMODE_WEIGHTED = 0x0002  # weight = 1/(1+error)

    def __init__(self):
        self.lib = None
        self.cached_buffer = {}
        self.cached_err_buffer = {}
        self.lib_lock = threading.Lock()
        self.cache_lock = threading.Lock()
        self.normalize_lock = threading.Lock()
        self.init_lib()

    def __call__(self, a: Config) -> Tuple[np.ndarray, np.ndarray]:
        # Validation checks
        if a.patch_size < 3:
            raise ValueError("Patch size is too small.")
        if a.patch_size % 2 == 0:
            raise ValueError("Patch size must be an odd number.")
        if len(a.guides) == 0:
            raise ValueError("At least one guide must be specified.")

        style_image = self._normalize_img_shape(_normalize_image(a.style_image))
        style_height, style_weight, style_channels = style_image.shape
        target_height, target_width, target_channels = 0, 0, 0

        if style_channels > self.MAX_STYLE_CHANNELS:
            raise ValueError(f"Too many style channels {style_channels}, maximum number is {self.MAX_STYLE_CHANNELS}.")

        guides_source = []
        guides_target = []
        guides_weights = []

        for source_guide, target_guide, guide_weight in a.guides:
            source_guide = self._normalize_img_shape(_normalize_image(source_guide))
            target_guide = self._normalize_img_shape(_normalize_image(target_guide))
            guide_weight = _normalize_weight(guide_weight)

            s_h, s_w, s_c = source_guide.shape
            t_h, t_w, t_c = target_guide.shape

            if s_h != style_height or s_w != style_weight:
                raise ValueError("Guide source and style resolution must match style resolution.")

            if target_channels == 0:
                target_height, target_width, target_channels = t_h, t_w, t_c
            elif t_h != target_height or t_w != target_width:
                raise ValueError("Guides target resolutions must be equal.")

            if s_c != t_c:
                raise ValueError("Guide source and target channels must match exactly.")

            guides_source.append(source_guide)
            guides_target.append(target_guide)

            guides_weights += [guide_weight / s_c] * s_c

        guides_source = np.concatenate(guides_source, axis = -1)
        guides_target = np.concatenate(guides_target, axis = -1)
        # noinspection PyCallingNonCallable,PyTypeChecker
        guides_weights = (c_float * len(guides_weights))(*guides_weights)

        style_weights = [1.0 / style_channels for _ in range(style_channels)]
        style_weights = (c_float * style_channels)(*style_weights)

        max_pyramid_levels = 0
        for level in range(32, -1, -1):
            if min(min(style_height, target_height) * pow(2.0, -level), min(style_weight, target_width) * pow(2.0, -level)) >= (2 * a.patch_size + 1):
                max_pyramid_levels = level + 1
                break

        if a.num_pyramid_levels == -1:
            num_pyramid_levels = max_pyramid_levels
        else:
            num_pyramid_levels = a.num_pyramid_levels
        num_pyramid_levels = min(num_pyramid_levels, max_pyramid_levels)

        # noinspection PyCallingNonCallable,PyTypeChecker
        num_search_vote_iters_per_level = (c_int * num_pyramid_levels)(*[a.num_search_vote_iters] * num_pyramid_levels)
        # noinspection PyCallingNonCallable,PyTypeChecker
        num_patch_match_iters_per_level = (c_int * num_pyramid_levels)(*[a.num_patch_match_iters] * num_pyramid_levels)
        # noinspection PyCallingNonCallable,PyTypeChecker
        stop_threshold_per_level = (c_int * num_pyramid_levels)(*[a.stop_threshold] * num_pyramid_levels)

        # Get or create buffers
        buffer = self._get_or_create_buffer((target_height, target_width, style_channels))
        err_buffer = self._get_or_create_err_buffer((target_height, target_width))

        print("Calling Ebsynth.")
        with self.lib_lock:
            self.lib.ebsynthRun(
                self.BACKEND_AUTO,  # backend
                style_channels,  # numStyleChannels
                guides_source.shape[-1],  # numGuideChannels
                style_weight,  # sourceWidth
                style_height,  # sourceHeight
                style_image.tobytes(),  # sourceStyleData (width * height * numStyleChannels) bytes, scan-line order
                guides_source.tobytes(),  # sourceGuideData (width * height * numGuideChannels) bytes, scan-line order
                target_width,  # targetWidth
                target_height,  # targetHeight
                guides_target.tobytes(),  # targetGuideData (width * height * numGuideChannels) bytes, scan-line order
                None,  # targetModulationData (width * height * numGuideChannels) bytes, scan-line order; pass NULL to switch off the modulation
                style_weights,  # styleWeights (numStyleChannels) floats
                guides_weights,  # guideWeights (numGuideChannels) floats
                a.uniformity,  # uniformityWeight reasonable values are between 500-15000, 3500 is a good default
                a.patch_size,  # patchSize odd sizes only, use 5 for 5x5 patch, 7 for 7x7, etc.
                self.VOTEMODE_WEIGHTED,  # voteMode use VOTEMODE_WEIGHTED for sharper result
                num_pyramid_levels,  # numPyramidLevels
                num_search_vote_iters_per_level,  # numSearchVoteItersPerLevel how many search/vote iters to perform at each level (array of ints, coarse first, fine last)
                num_patch_match_iters_per_level,  # numPatchMatchItersPerLevel how many Patch-Match iters to perform at each level (array of ints, coarse first, fine last)
                stop_threshold_per_level,  # stopThresholdPerLevel stop improving pixel when its change since last iteration falls under this threshold
                1 if a.extra_pass3x3 else 0,  # extraPass3x3 perform additional polishing pass with 3x3 patches at the finest level, use 0 to disable
                None,  # outputNnfData (width * height * 2) ints, scan-line order; pass NULL to ignore
                buffer,  # outputImageData  (width * height * numStyleChannels) bytes, scan-line order
                err_buffer,  # outputErrorData (width * height) floats, scan-line order; pass NULL to ignore
            )

        img = np.frombuffer(buffer, dtype = np.uint8).reshape((target_height, target_width, style_channels)).copy()
        err = np.frombuffer(err_buffer, dtype = np.float32).reshape((target_height, target_width)).copy()

        return img, err

    def init_lib(self):
        with self.lib_lock:
            if self.lib is None:
                if sys.platform[0:3] == "win":
                    self.lib = CDLL(os.path.join(Path(__file__).parent, "utils", "ebsynth.dll"))
                elif sys.platform == "darwin":
                    self.lib = CDLL(os.path.join(Path(__file__).parent, "utils", "ebsynth.so"))
                elif sys.platform[0:5] == "linux":
                    self.lib = CDLL(os.path.join(Path(__file__).parent, "utils", "ebsynth.so"))

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

    def _get_or_create_buffer(self, key):
        with self.cache_lock:
            a = self.cached_buffer.get(key, None)
            if a is None:
                a = create_string_buffer(key[0] * key[1] * key[2])
                self.cached_buffer[key] = a
            return a

    def _get_or_create_err_buffer(self, key):
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


def _normalize_image(a: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(a, str):
        b = cv2.imread(a)
        if b is None:
            raise ValueError("Cannot read image: " + str(a))
        return b

    elif isinstance(a, np.ndarray):
        if a.shape[-1] != 3:
            raise ValueError("Image a 3-channel numpy array.")
        return a

    else:
        raise ValueError("Image must valid file path or a 3-channel numpy array.")


def _normalize_weight(a: Union[int, float, None]) -> float:
    if isinstance(a, int):
        return float(a)

    elif isinstance(a, float):
        return a

    elif a is None:
        return 1.0

    else:
        raise ValueError("Weight should be int or float or none.")
