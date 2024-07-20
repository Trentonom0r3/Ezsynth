import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)


class Warp:
    def __init__(self, img):
        # self.lock = threading.Lock()
        H, W, _ = img.shape
        self.H = H
        self.W = W
        self.grid = self._create_grid(H, W)

    def _create_grid(self, H: int, W: int):
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        return np.stack((x, y), axis=-1).astype(np.float32)

    def _warp(self, img: np.ndarray, flo: np.ndarray):
        # with self.lock:
        flo_resized = cv2.resize(flo, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        map_x = self.grid[..., 0] + flo_resized[..., 0]
        map_y = self.grid[..., 1] + flo_resized[..., 1]
        warped_img = cv2.remap(
            img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return warped_img

    def run_warping(self, img: np.ndarray, flow: np.ndarray):
        img = img.astype(np.float32)
        flow = flow.astype(np.float32)

        try:
            warped_img = self._warp(img, flow)
            warped_image = (warped_img * 255).astype(np.uint8)
            return warped_image
        except Exception as e:
            print(f"[ERROR] Exception in run_warping: {e}")
            return None
