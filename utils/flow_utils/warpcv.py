import numpy as np
import cv2
import cProfile

class WarpCV:
    def __init__(self, img):
        H, W, _ = img.shape
        self.H = H
        self.W = W
        self.grid = self._create_grid(H, W)
    
    def _create_grid(self, H, W):
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        return np.stack((x, y), axis=-1).astype(np.float32)
    
    def _warp(self, img, flo):
        flo_resized = cv2.resize(flo, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        map_x = self.grid[..., 0] + flo_resized[..., 0]
        map_y = self.grid[..., 1] + flo_resized[..., 1]
        warped_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return warped_img

    def run_warping(self, img, flow):
        img = img.astype(np.float32)
        img = flow.astype(np.float32)
        
        try:
            warped_img = self._warp(img, flow)
            warped_image = warped_img.astype(np.uint8)
            return warped_img
        except Exception as e:
            print(f"[ERROR] Exception in run_warping: {e}")
            return None

if __name__ == '__main__':
    image = np.random.rand(540, 960, 3).astype(np.float32)
    flow = np.random.rand(540, 960, 2).astype(np.float32)
    warp_cv = WarpCV(image)
    warped_image = warp_cv.run_warping(image, flow)
    cProfile.run('warp_cv.run_warping(image, flow)')
