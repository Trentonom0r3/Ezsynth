import cv2
import numpy as np
from ezsynth.utils.flow_utils.warp import Warp


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
