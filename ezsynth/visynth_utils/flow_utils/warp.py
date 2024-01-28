import cv2
import numpy as np


class Warp:
    def __init__(self, img):
        # self.lock = threading.Lock()
        height, width, _ = img.shape
        self.H = height
        self.W = width
        self.grid = self._create_grid(height, width)

    def _create_grid(self, H, W):
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        return np.stack((x, y), axis = -1).astype(np.float32)

    def _warp(self, img, flo):
        # with self.lock:
        flo_resized = cv2.resize(flo, (self.W, self.H), interpolation = cv2.INTER_LINEAR)
        map_x = self.grid[..., 0] + flo_resized[..., 0]
        map_y = self.grid[..., 1] + flo_resized[..., 1]
        warped_img = cv2.remap(img, map_x, map_y, interpolation = cv2.INTER_LINEAR, borderMode = cv2.BORDER_REFLECT)
        return warped_img

    def run_warping(self, img, flow):
        img = img.astype(np.float32)
        flow = flow.astype(np.float32)

        try:
            warped_img = self._warp(img, flow)
            warped_image = (warped_img * 255).astype(np.uint8)
            return warped_image
        except Exception as e:
            print(f"[ERROR] Exception in run_warping: {e}")
            return None
