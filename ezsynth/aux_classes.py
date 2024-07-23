import cv2
import numpy as np
from ezsynth.utils.flow_utils.warp import Warp
from ezsynth.utils.sequences import EasySequence


class RunConfig:
    def __init__(
        self,
        edg_wgt=1.0,
        img_wgt=6.0,
        pos_wgt=2.0,
        wrp_wgt=0.5,
        uniformity=3500.0,
        patchsize=7,
        pyramidlevels=6,
        searchvoteiters=12,
        patchmatchiters=6,
        extrapass3x3=True,
        use_gpu=False,
        use_lsqr=True,
        use_poisson_cupy=False,
        poisson_maxiter=None,
        only_mode=EasySequence.MODE_NON
    ) -> None:
        # Weights
        self.edg_wgt = edg_wgt
        self.img_wgt = img_wgt
        self.pos_wgt = pos_wgt
        self.wrp_wgt = wrp_wgt

        # Ebsynth gen params
        self.uniformity = uniformity
        self.patchsize = patchsize
        self.pyramidlevels = pyramidlevels
        self.searchvoteiters = searchvoteiters
        self.patchmatchiters = patchmatchiters
        self.extrapass3x3 = extrapass3x3

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
        coord_map = np.zeros((h, w, 3), dtype=np.float32)
        coord_map[:, :, 0] = np.linspace(0, 1, w)
        coord_map[:, :, 1] = np.linspace(0, 1, h)[:, np.newaxis]
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
