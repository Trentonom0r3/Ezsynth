import cv2
import numpy as np


def hist_blender(
    a: np.ndarray,
    b: np.ndarray,
    error_mask: np.ndarray,
    weight1=0.5,
    weight2=0.5,
) -> np.ndarray:
    # Ensure error_mask has 3 channels
    if len(error_mask.shape) == 2:
        error_mask = np.repeat(error_mask[:, :, np.newaxis], 3, axis=2)

    # Convert to Lab color space
    a_lab = cv2.cvtColor(a, cv2.COLOR_BGR2Lab)
    b_lab = cv2.cvtColor(b, cv2.COLOR_BGR2Lab)

    # Generate min_error_img
    min_error_lab = assemble_min_error_img(a_lab, b_lab, error_mask)

    # Compute means and stds
    a_mean, a_std = mean_std(a_lab)
    b_mean, b_std = mean_std(b_lab)
    min_error_mean, min_error_std = mean_std(min_error_lab)

    # Histogram transformation constants
    t_mean = np.full(3, 0.5 * 256, dtype=np.float32)
    t_std = np.full(3, (1 / 36) * 256, dtype=np.float32)

    # Histogram transform
    a_lab = histogram_transform(a_lab, a_mean, a_std, t_mean, t_std)
    b_lab = histogram_transform(b_lab, b_mean, b_std, t_mean, t_std)

    # Blending
    ab_lab = (a_lab * weight1 + b_lab * weight2 - 128) / 0.5 + 128
    ab_mean, ab_std = mean_std(ab_lab)

    # Final histogram transform
    ab_lab = histogram_transform(ab_lab, ab_mean, ab_std, min_error_mean, min_error_std)

    ab_lab = np.clip(np.round(ab_lab), 0, 255).astype(np.uint8)

    # Convert back to BGR
    ab = cv2.cvtColor(ab_lab, cv2.COLOR_Lab2BGR)

    return ab


def histogram_transform(img, means, stds, target_means, target_stds):
    return ((img - means) * target_stds / stds + target_means).astype(np.float32)


def assemble_min_error_img(a, b, error_mask):
    return np.where(error_mask == 0, a, b)


def mean_std(img):
    return np.mean(img, axis=(0, 1)), np.std(img, axis=(0, 1))
