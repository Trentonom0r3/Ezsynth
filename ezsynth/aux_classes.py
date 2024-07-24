import cv2
import numpy as np

from ezsynth.utils.flow_utils.warp import Warp
from ezsynth.utils.sequences import EasySequence


class RunConfig:
    def __init__(
        self,
        uniformity=3500.0,
        patchsize=7,
        pyramidlevels=6,
        searchvoteiters=12,
        patchmatchiters=6,
        extrapass3x3=True,
        edg_wgt=1.0,
        img_wgt=6.0,
        pos_wgt=2.0,
        wrp_wgt=0.5,
        use_gpu=False,
        use_lsqr=True,
        use_poisson_cupy=False,
        poisson_maxiter=None,
        only_mode=EasySequence.MODE_NON,
    ) -> None:
        # Ebsynth gen params
        self.uniformity = uniformity
        self.patchsize = patchsize
        self.pyramidlevels = pyramidlevels
        self.searchvoteiters = searchvoteiters
        self.patchmatchiters = patchmatchiters
        self.extrapass3x3 = extrapass3x3

        # Weights
        self.edg_wgt = edg_wgt
        self.img_wgt = img_wgt
        self.pos_wgt = pos_wgt
        self.wrp_wgt = wrp_wgt

        # Blend params
        self.use_gpu = use_gpu
        self.use_lsqr = use_lsqr
        self.use_poisson_cupy = use_poisson_cupy
        self.poisson_maxiter = poisson_maxiter

        # No blending mode
        self.only_mode = only_mode

        # Skip adding last style frame if blending
        self.skip_blend_style_last = False

    def get_ebsynth_cfg(self):
        return {
            "uniformity": self.uniformity,
            "patchsize": self.patchsize,
            "pyramidlevels": self.pyramidlevels,
            "searchvoteiters": self.searchvoteiters,
            "patchmatchiters": self.patchmatchiters,
            "extrapass3x3": self.extrapass3x3,
        }

    def get_blender_cfg(self):
        return {
            "use_gpu": self.use_gpu,
            "use_lsqr": self.use_lsqr,
            "use_poisson_cupy": self.use_poisson_cupy,
            "poisson_maxiter": self.poisson_maxiter,
        }


class PositionalGuide:
    def __init__(self) -> None:
        self.coord_map = None
        self.coord_map_warped = None

    def get_coord_maps(self, warp: Warp):
        h, w = warp.H, warp.W

        # Create x and y coordinates
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)

        # Use numpy's meshgrid to create 2D coordinate arrays
        xx, yy = np.meshgrid(x, y)

        # Stack the coordinates into a single 3D array
        coord_map = np.stack((xx, yy, np.zeros_like(xx)), axis=-1).astype(np.float32)

        coord_map_warped = coord_map.copy()

        self.coord_map = coord_map
        self.coord_map_warped = coord_map_warped

    def get_or_create_coord_maps(self, warp: Warp):
        if self.coord_map is None or self.coord_map_warped is None:
            self.get_coord_maps(warp)
        return self.coord_map, self.coord_map_warped

    def create_from_flow(
        self, flow: np.ndarray, original_size: tuple[int, ...], warp: Warp
    ):
        coord_map, coord_map_warped = self.get_or_create_coord_maps(warp)
        coord_map_warped = warp.run_warping(coord_map, flow)
        g_pos = cv2.resize(coord_map_warped, original_size)
        g_pos = np.clip(g_pos, 0, 1)
        g_pos = (g_pos * 255).astype(np.uint8)
        self.coord_map = self.coord_map_warped.copy()
        return g_pos


class EdgeConfig:
    # PST
    PST_S = 0.3
    PST_W = 15
    PST_SIG_LPF = 0.15
    PST_MIN = 0.05
    PST_MAX = 0.9

    # PAGE
    PAGE_M1 = 0
    PAGE_M2 = 0.35
    PAGE_SIG1 = 0.05
    PAGE_SIG2 = 0.8
    PAGE_S1 = 0.8
    PAGE_S2 = 0.8
    PAGE_SIG_LPF = 0.1
    PAGE_MIN = 0.0
    PAGE_MAX = 0.9

    MORPH_FLAG = 1

    def __init__(self, **kwargs):
        # PST attributes
        self.pst_s = kwargs.get("S", self.PST_S)
        self.pst_w = kwargs.get("W", self.PST_W)
        self.pst_sigma_lpf = kwargs.get("sigma_LPF", self.PST_SIG_LPF)
        self.pst_thresh_min = kwargs.get("thresh_min", self.PST_MIN)
        self.pst_thresh_max = kwargs.get("thresh_max", self.PST_MAX)

        # PAGE attributes
        self.page_mu_1 = kwargs.get("mu_1", self.PAGE_M1)
        self.page_mu_2 = kwargs.get("mu_2", self.PAGE_M2)
        self.page_sigma_1 = kwargs.get("sigma_1", self.PAGE_SIG1)
        self.page_sigma_2 = kwargs.get("sigma_2", self.PAGE_SIG2)
        self.page_s1 = kwargs.get("S1", self.PAGE_S1)
        self.page_s2 = kwargs.get("S2", self.PAGE_S2)
        self.page_sigma_lpf = kwargs.get("sigma_LPF", self.PAGE_SIG_LPF)
        self.page_thresh_min = kwargs.get("thresh_min", self.PAGE_MIN)
        self.page_thresh_max = kwargs.get("thresh_max", self.PAGE_MAX)

        self.morph_flag = kwargs.get("morph_flag", self.MORPH_FLAG)

    @classmethod
    def get_pst_default(cls) -> dict:
        return {
            "S": cls.PST_S,
            "W": cls.PST_W,
            "sigma_LPF": cls.PST_SIG_LPF,
            "thresh_min": cls.PST_MIN,
            "thresh_max": cls.PST_MAX,
            "morph_flag": cls.MORPH_FLAG,
        }

    @classmethod
    def get_page_default(cls) -> dict:
        return {
            "mu_1": cls.PAGE_M1,
            "mu_2": cls.PAGE_M2,
            "sigma_1": cls.PAGE_SIG1,
            "sigma_2": cls.PAGE_SIG2,
            "S1": cls.PAGE_S1,
            "S2": cls.PAGE_S2,
            "sigma_LPF": cls.PAGE_SIG_LPF,
            "thresh_min": cls.PAGE_MIN,
            "thresh_max": cls.PAGE_MAX,
            "morph_flag": cls.MORPH_FLAG,
        }

    def get_pst_current(self) -> dict:
        return {
            "S": self.pst_s,
            "W": self.pst_w,
            "sigma_LPF": self.pst_sigma_lpf,
            "thresh_min": self.pst_thresh_min,
            "thresh_max": self.pst_thresh_max,
            "morph_flag": self.morph_flag,
        }

    def get_page_current(self) -> dict:
        return {
            "mu_1": self.page_mu_1,
            "mu_2": self.page_mu_2,
            "sigma_1": self.page_sigma_1,
            "sigma_2": self.page_sigma_2,
            "S1": self.page_s1,
            "S2": self.page_s2,
            "sigma_LPF": self.page_sigma_lpf,
            "thresh_min": self.page_thresh_min,
            "thresh_max": self.page_thresh_max,
            "morph_flag": self.morph_flag,
        }
