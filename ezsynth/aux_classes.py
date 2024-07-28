import cv2
import numpy as np

from .utils.flow_utils.warp import Warp
from .sequences import EasySequence


class RunConfig:
    """
    ### Ebsynth gen params
        `uniformity (float)`: Uniformity weight for the style transfer.
        Reasonable values are between `500-15000`.
        Defaults to `3500.0`.

        `patchsize (int)`: Size of the patches [NxN]. Must be an odd number `>= 3`.
        Defaults to `7`.

        `pyramidlevels (int)`: Number of pyramid levels. Larger values useful for things like color transfer.
        Defaults to `6`.

        `searchvoteiters (int)`: Number of search/vote iterations. Defaults to `12`.
        `patchmatchiters (int)`: Number of Patch-Match iterations. The larger, the longer it takes.
        Defaults to `6`.

        `extrapass3x3 (bool)`: Perform additional polishing pass with 3x3 patches at the finest level.
        Defaults to `True`.

    ### Ebsynth guide weights params
        `edg_wgt (float)`: Edge detect weights. Defaults to `1.0`.
        `img_wgt (float)`: Original image weights. Defaults to `6.0`.
        `pos_wgt (float)`: Flow position warping weights. Defaults to `2.0`.
        `wrp_wgt (float)`: Warped style image weight. Defaults to `0.5`.

    ### Blending params
        `use_gpu (bool)`: Use GPU for Histogram Blending (Only affect Blend mode). Faster than CPU.
        Defaults to `False`.

        `use_lsqr (bool)`: Use LSQR (Least-squares solver) instead of LSMR (Iterative solver for least-squares)
        for Poisson blending step. LSQR often yield better results. May change to LSMR for speed (depends).
        Defaults to `True`.

        `use_poisson_cupy (bool)`: Use Cupy GPU acceleration for Poisson blending step.
        Uses LSMR (overrides `use_lsqr`). May not yield better speed.
        Defaults to `False`.

        `poisson_maxiter (int | None)`: Max iteration to calculate Poisson Least-squares (only affect LSMR mode).
        Expect positive integers.
        Defaults to `None`.

        `only_mode (str)`: Skip blending, only run one pass per sequence.
            Valid values:
                `MODE_FWD = "forward"` (Will only run forward mode if `sequence.mode` is blend)

                `MODE_REV = "reverse"` (Will only run reverse mode if `sequence.mode` is blend)

            Defaults to `MODE_NON = "none"`.

    ### Masking params
        `do_mask (bool)`: Whether to apply mask. Defaults to `False`.

        `pre_mask (bool)`: Whether to mask the inputs and styles before `RUN` or after.
        Pre-mask takes ~2x time to run per frame. Could be due to Ebsynth.dll implementation.
        Defaults to `False`.

        `feather (int)`: Feather Gaussian radius to apply on the mask results. Only affect if `return_masked_only == False`.
        Expects integers. Defaults to `0`.
    """

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
        poisson_maxiter: int | None = None,
        only_mode=EasySequence.MODE_NON,
        do_mask=False,
        pre_mask=False,
        feather=0,
    ) -> None:
        # Ebsynth gen params
        self.uniformity = uniformity
        """Uniformity weight for the style transfer.
        Reasonable values are between `500-15000`.

        Defaults to `3500.0`."""
        self.patchsize = patchsize
        """Size of the patches [`NxN`]. Must be an odd number `>= 3`.
        Defaults to `7`"""

        self.pyramidlevels = pyramidlevels
        """Number of pyramid levels.
        Larger values useful for things like color transfer.

        Defaults to 6."""

        self.searchvoteiters = searchvoteiters
        """Number of search/vote iterations.
        Defaults to `12`"""

        self.patchmatchiters = patchmatchiters
        """Number of Patch-Match iterations. The larger, the longer it takes.
        Defaults to `6`"""

        self.extrapass3x3 = extrapass3x3
        """Perform additional polishing pass with 3x3 patches at the finest level.
        Defaults to `True`"""

        # Weights
        self.edg_wgt = edg_wgt
        """Edge detect weights. Defaults to `1.0`"""

        self.img_wgt = img_wgt
        """Original image weights. Defaults to `6.0`"""

        self.pos_wgt = pos_wgt
        """Flow position warping weights. Defaults to `2.0`"""

        self.wrp_wgt = wrp_wgt
        """Warped style image weight. Defaults to `0.5`"""

        # Blend params
        self.use_gpu = use_gpu
        """Use GPU for Histogram Blending (Only affect Blend mode). Faster than CPU.
        Defaults to `False`"""

        self.use_lsqr = use_lsqr
        """Use LSQR (Least-squares solver) instead of LSMR (Iterative solver for least-squares)
        for Poisson blending step. LSQR often yield better results.

        May change to LSMR for speed (depends).
        Defaults to `True`"""

        self.use_poisson_cupy = use_poisson_cupy
        """Use Cupy GPU acceleration for Poisson blending step.
        Uses LSMR (overrides `use_lsqr`). May not yield better speed.

        Defaults to `False`"""

        self.poisson_maxiter = poisson_maxiter
        """Max iteration to calculate Poisson Least-squares (only affect LSMR mode). Expect positive integers.

        Defaults to `None`"""

        # No blending mode
        self.only_mode = only_mode
        """Skip blending, only run one pass per sequence.

        Valid values:
            `MODE_FWD = "forward"` (Will only run forward mode if `sequence.mode` is blend)

            `MODE_REV = "reverse"` (Will only run reverse mode if `sequence.mode` is blend)

        Defaults to `MODE_NON = "none"`
        """

        # Skip adding last style frame if blending
        self.skip_blend_style_last = False
        """Skip adding last style frame if blending. Internal variable"""

        # Masking mode
        self.do_mask = do_mask
        """Whether to apply mask. Defaults to `False`"""

        self.pre_mask = pre_mask
        """Whether to mask the inputs and styles before `RUN` or after.

        Pre-mask takes ~2x time to run per frame. Could be due to Ebsynth.dll implementation.

        Defaults to `False`"""

        self.feather = feather
        """Feather Gaussian radius to apply on the mask results. Only affect if `return_masked_only == False`.

        Expects integers. Defaults to `0`"""

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

    def get_coord_maps(self, warp: Warp):
        h, w = warp.H, warp.W

        # Create x and y coordinates
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)

        # Use numpy's meshgrid to create 2D coordinate arrays
        xx, yy = np.meshgrid(x, y)

        # Stack the coordinates into a single 3D array
        self.coord_map = np.stack((xx, yy, np.zeros_like(xx)), axis=-1).astype(
            np.float32
        )

    def get_or_create_coord_maps(self, warp: Warp):
        if self.coord_map is None is None:
            self.get_coord_maps(warp)
        return self.coord_map

    def create_from_flow(
        self, flow: np.ndarray, original_size: tuple[int, ...], warp: Warp
    ):
        coord_map = self.get_or_create_coord_maps(warp)
        coord_map_warped = warp.run_warping(coord_map, flow)

        coord_map_warped[..., :2] = coord_map_warped[..., :2] % 1

        if coord_map_warped.shape[:2] != original_size:
            coord_map_warped = cv2.resize(
                coord_map_warped, original_size, interpolation=cv2.INTER_LINEAR
            )

        g_pos = (coord_map_warped * 255).astype(np.uint8)
        self.coord_map = coord_map_warped.copy()

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
