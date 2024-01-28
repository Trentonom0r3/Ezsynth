from typing import Literal

import cv2
import numpy as np
import phycv
import torch


class EdgeDetector:
    def __init__(
            self,
            method: Literal["PAGE", "PST", "Classic"] = "PAGE",
            device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the edge detector.

        :param method: Edge detection method.
            PAGE - Phase and Gradient Estimation
                Great detail, great structure, but slow.
            PST - Phase Stretch Transform
                Good overall structure, but not very detailed.
            Classic
                A good balance between structure and detail.
        :param device: What processing unit to use.
        """
        self.method = method
        self.device = device

        if method == "PAGE":
            self.page_gpu = phycv.PAGE_GPU(direction_bins = 10, device = self.device)

        elif method == "PST":
            self.pst_gpu = phycv.PST_GPU(device = self.device)

        elif method == "Classic":
            self.kernel = _create_gaussian_kernel(5, 6.0)

        else:
            raise ValueError("Unknown edge detection method.")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Compute the edge map.

        :param image: A numpy array.

        :return: Edge map as a numpy array.
        """
        if self.method == "PAGE":
            mu_1, mu_2, sigma_1, sigma_2, S1, S2, sigma_LPF, thresh_min, thresh_max, morph_flag = 0, 0.35, 0.05, 0.8, 0.8, 0.8, 0.1, 0.0, 0.9, True
            self.page_gpu.load_img(img_array = image)
            self.page_gpu.init_kernel(mu_1, mu_2, sigma_1, sigma_2, S1, S2)
            self.page_gpu.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)
            self.page_gpu.create_page_edge()
            result = self.page_gpu.page_edge
            result = result.cpu().numpy()
            result = cv2.GaussianBlur(result, (5, 5), 3)
            result = (result * 255).astype(np.uint8)

        elif self.method == "PST":
            S, W, sigma_LPF, thresh_min, thresh_max, morph_flag = 0.3, 15, 0.15, 0.05, 0.9, True
            result = self.pst_gpu.pst_output
            result = result.cpu().numpy()
            result = cv2.GaussianBlur(result, (5, 5), 3)
            result = (result * 255).astype(np.uint8)

        elif self.method == "Classic":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.filter2D(gray, -1, self.kernel)
            result = cv2.subtract(gray, blurred)
            result = cv2.add(result, 0.5 * 255)
            result = np.clip(result, 0, 255)

        else:
            raise ValueError("Unknown edge detection method.")

        return result


def _create_gaussian_kernel(size, sigma):
    """
    Create a Gaussian kernel.

    """
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size))

    return kernel / np.sum(kernel)
