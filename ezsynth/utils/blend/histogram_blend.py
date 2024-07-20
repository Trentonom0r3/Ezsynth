import cv2
import numpy as np
import torch
import kornia as K
import kornia.color as KC

try:
    import cupy as cp
except:   # noqa: E722
    print("Cupy is not installed. Will use normal Numpy for Histogram blending")
    USE_GPU = False

class HistogramBlender:
    def __init__(self, use_gpu=True):
        
        self.xp = cp if use_gpu and USE_GPU else np
        self.use_gpu = use_gpu and USE_GPU

    def histogram_transform(self, img, means, stds, target_means, target_stds):
        return ((img - means) * target_stds / stds + target_means).astype(self.xp.float32)

    def assemble_min_error_img(self, a, b, error_mask):
        return self.xp.where(error_mask == 0, a, b)

    def mean_std(self, img):
        return self.xp.mean(img, axis=(0, 1)), self.xp.std(img, axis=(0, 1))

    def blend(self, a, b, error_mask, weight1=0.5, weight2=0.5):
        if self.use_gpu:
            a = cp.asarray(a)
            b = cp.asarray(b)
            error_mask = cp.asarray(error_mask)

        # Ensure error_mask has 3 channels
        if len(error_mask.shape) == 2:
            error_mask = self.xp.repeat(error_mask[:, :, self.xp.newaxis], 3, axis=2)

        # Convert to Lab color space
        a_lab = cv2.cvtColor(self.xp.asnumpy(a) if self.use_gpu else a, cv2.COLOR_BGR2Lab)
        b_lab = cv2.cvtColor(self.xp.asnumpy(b) if self.use_gpu else b, cv2.COLOR_BGR2Lab)
        
        if self.use_gpu:
            a_lab = cp.asarray(a_lab)
            b_lab = cp.asarray(b_lab)

        # Generate min_error_img
        min_error_lab = self.assemble_min_error_img(a_lab, b_lab, error_mask)

        # Compute means and stds
        a_mean, a_std = self.mean_std(a_lab)
        b_mean, b_std = self.mean_std(b_lab)
        min_error_mean, min_error_std = self.mean_std(min_error_lab)

        # Histogram transformation constants
        t_mean = self.xp.full(3, 0.5 * 256, dtype=self.xp.float32)
        t_std = self.xp.full(3, (1 / 36) * 256, dtype=self.xp.float32)

        # Histogram transform
        a_lab = self.histogram_transform(a_lab, a_mean, a_std, t_mean, t_std)
        b_lab = self.histogram_transform(b_lab, b_mean, b_std, t_mean, t_std)

        # Blending
        ab_lab = (a_lab * weight1 + b_lab * weight2 - 128) / 0.5 + 128
        ab_mean, ab_std = self.mean_std(ab_lab)

        # Final histogram transform
        ab_lab = self.histogram_transform(ab_lab, ab_mean, ab_std, min_error_mean, min_error_std)
        
        ab_lab = self.xp.clip(self.xp.round(ab_lab), 0, 255).astype(self.xp.uint8)

        # Convert back to BGR
        if self.use_gpu:
            ab_lab = cp.asnumpy(ab_lab)
        ab = cv2.cvtColor(ab_lab, cv2.COLOR_Lab2BGR)

        return ab
    
# class HistogramBlender:
#     def __init__(self):
#         pass

#     def histogram_transform(
#         self,
#         img: np.ndarray,
#         means: np.ndarray,
#         stds: np.ndarray,
#         target_means: np.ndarray,
#         target_stds: np.ndarray,
#     ):
#         st = time.time()
#         means = means.reshape((1, 1, 3))
#         stds = stds.reshape((1, 1, 3))
#         target_means = target_means.reshape((1, 1, 3))
#         target_stds = target_stds.reshape((1, 1, 3))
#         x = img.astype(np.float32)
#         x = (x - means) * target_stds / stds + target_means
#         print(f"Hist trans took {time.time() - st:.4f} s")
#         return x

#     def assemble_min_error_img(
#         self, a: np.ndarray, b: np.ndarray, error_mask: np.ndarray
#     ) -> np.ndarray:
#         min_error_img = np.where(error_mask == 0, a, b)
#         return min_error_img

#     def bgr_to_lab(self, img: np.ndarray):
#         return cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

#     def mean_std(self, img: np.ndarray):
#         mean = np.mean(img, axis=(0, 1))
#         std = np.std(img, axis=(0, 1))
#         return mean, std

#     def blend(
#         self,
#         a: np.ndarray,
#         b: np.ndarray,
#         error_mask: np.ndarray,
#         weight1=0.5,
#         weight2=0.5,
#     ):
#         # Ensure error_mask has 3 channels
#         if len(error_mask.shape) == 2:
#             error_mask = np.repeat(error_mask[:, :, np.newaxis], 3, axis=2)

#         # Generate min_error_img
#         min_error = self.assemble_min_error_img(a, b, error_mask)

#         # Convert to Lab color space
#         a_lab = self.bgr_to_lab(a)
#         b_lab = self.bgr_to_lab(b)
#         min_error_lab = self.bgr_to_lab(min_error)

#         # Compute means and stds
#         a_mean, a_std = self.mean_std(a_lab)
#         b_mean, b_std = self.mean_std(b_lab)
#         min_error_mean, min_error_std = self.mean_std(min_error_lab)

#         # Histogram transformation constants
#         t_mean_val = 0.5 * 256
#         t_std_val = (1 / 36) * 256
#         t_mean = np.ones([3], dtype=np.float32) * t_mean_val
#         t_std = np.ones([3], dtype=np.float32) * t_std_val

#         # Histogram transform
#         a_lab = self.histogram_transform(a_lab, a_mean, a_std, t_mean, t_std)
#         b_lab = self.histogram_transform(b_lab, b_mean, b_std, t_mean, t_std)

#         # Blending
#         ab_lab = (a_lab * weight1 + b_lab * weight2 - t_mean_val) / 0.5 + t_mean_val
#         ab_mean, ab_std = self.mean_std(ab_lab)

#         # Final histogram transform
#         ab_lab = self.histogram_transform(
#             ab_lab, ab_mean, ab_std, min_error_mean, min_error_std
#         )
        
#         ab_lab = np.clip(np.round(ab_lab), 0, 255).astype(np.uint8)

#         # Convert back to BGR
#         ab = cv2.cvtColor(ab_lab, cv2.COLOR_Lab2BGR)

#         return ab
