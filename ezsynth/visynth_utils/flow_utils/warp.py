import cv2
import numpy as np


class Warp:
    def __init__(self, image):
        # self.lock = threading.Lock()
        height, width, _ = image.shape
        self.height = height
        self.width = width
        self.grid = self._create_grid(height, width)

    # noinspection PyMethodMayBeStatic
    def _create_grid(self, height, width):
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        return np.stack((x, y), axis = -1).astype(np.float32)

    def _warp(self, image, flo):
        # with self.lock:
        flo_resized = cv2.resize(flo, (self.width, self.height), interpolation = cv2.INTER_LINEAR)
        map_x = self.grid[..., 0] + flo_resized[..., 0]
        map_y = self.grid[..., 1] + flo_resized[..., 1]
        warped_img = cv2.remap(image, map_x, map_y, interpolation = cv2.INTER_LINEAR, borderMode = cv2.BORDER_REFLECT)
        return warped_img

    def run_warping(self, img, flow):
        img = img.astype(np.float32)
        flow = flow.astype(np.float32)

        try:
            warped_img = self._warp(img, flow)
            warped_image = (warped_img * 255).astype(np.uint8)
            return warped_image
        except Exception as e:
            print(f"Exception in run_warping: {e}")
            return None
