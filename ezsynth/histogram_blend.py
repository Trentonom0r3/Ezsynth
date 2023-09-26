import cv2
import numpy as np

class HistogramBlender:
    def __init__(self):
        pass
    
    def histogram_transform(self, img, means, stds, target_means, target_stds):
        means = means.reshape((1, 1, 3))
        stds = stds.reshape((1, 1, 3))
        target_means = target_means.reshape((1, 1, 3))
        target_stds = target_stds.reshape((1, 1, 3))
        x = img.astype(np.float32)
        x = (x - means) * target_stds / stds + target_means
        return x
    
    def assemble_min_error_img(self, a, b, error_mask):
        min_error_img = b.copy()
        min_error_img[error_mask == 0] = a[error_mask == 0]
        min_error_img[error_mask == 1] = b[error_mask == 1]
        return min_error_img
    
    def blend(self, a, b, error_mask, weight1=0.5, weight2=0.5):
        # Generate min_error_img
        min_error = self.assemble_min_error_img(a, b, error_mask)
        
        # Convert to Lab color space
        a = cv2.cvtColor(a, cv2.COLOR_BGR2Lab)
        b = cv2.cvtColor(b, cv2.COLOR_BGR2Lab)
        min_error = cv2.cvtColor(min_error, cv2.COLOR_BGR2Lab)
        
        # Compute means and stds
        a_mean = np.mean(a, axis=(0, 1))
        a_std = np.std(a, axis=(0, 1))
        b_mean = np.mean(b, axis=(0, 1))
        b_std = np.std(b, axis=(0, 1))
        min_error_mean = np.mean(min_error, axis=(0, 1))
        min_error_std = np.std(min_error, axis=(0, 1))
        
        # Histogram transformation constants
        t_mean_val = 0.5 * 256
        t_std_val = (1 / 36) * 256
        t_mean = np.ones([3], dtype=np.float32) * t_mean_val
        t_std = np.ones([3], dtype=np.float32) * t_std_val
        
        # Histogram transform
        a = self.histogram_transform(a, a_mean, a_std, t_mean, t_std)
        b = self.histogram_transform(b, b_mean, b_std, t_mean, t_std)
        
        # Blending
        ab = (a * weight1 + b * weight2 - t_mean_val) / 0.5 + t_mean_val
        ab_mean = np.mean(ab, axis=(0, 1))
        ab_std = np.std(ab, axis=(0, 1))
        
        # Final histogram transform
        ab = self.histogram_transform(ab, ab_mean, ab_std, min_error_mean, min_error_std)
        ab = np.round(ab)
        ab = np.clip(ab, 0, 255)
        ab = ab.astype(np.uint8)
        
        # Convert back to BGR
        ab = cv2.cvtColor(ab, cv2.COLOR_Lab2BGR)
        
        return ab
