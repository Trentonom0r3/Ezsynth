import os
import tempfile
from typing import Literal

import cv2
import numpy as np
import phycv
import torch
from PIL import Image


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

    def compute_edge(self, input_data):
        """
        Compute the edge map.

        :param input_data: Either a file path or a numpy array.

        :return: Edge map as a numpy array.
        """
        if self.method == "PAGE":
            input_data_path = _load_image(input_data)
            try:
                # page_gpu = PAGE_GPU(direction_bins=10, device=self.device)
                mu_1, mu_2, sigma_1, sigma_2, S1, S2, sigma_LPF, thresh_min, thresh_max, morph_flag = 0, 0.35, 0.05, 0.8, 0.8, 0.8, 0.1, 0.0, 0.9, 1
                result = self.page_gpu.run(input_data_path, mu_1, mu_2, sigma_1, sigma_2, S1, S2, sigma_LPF, thresh_min, thresh_max, morph_flag)
                result = result.cpu().numpy()
                result = cv2.GaussianBlur(result, (5, 5), 3)
                result = (result * 255).astype(np.uint8)
            finally:
                os.remove(input_data_path)

        elif self.method == "PST":
            input_data_path = _load_image(input_data)
            try:
                # pst_gpu = PST_GPU(device=self.device)
                S, W, sigma_LPF, thresh_min, thresh_max, morph_flag = 0.3, 15, 0.15, 0.05, 0.9, 1
                result = self.pst_gpu.run(input_data_path, S, W, sigma_LPF, thresh_min, thresh_max, morph_flag)
                result = result.cpu().numpy()
                result = cv2.GaussianBlur(result, (5, 5), 3)
                result = (result * 255).astype(np.uint8)
            finally:
                os.remove(input_data_path)

        elif self.method == "Classic":
            if isinstance(input_data, np.ndarray):
                img = input_data
            elif isinstance(input_data, str):
                img = cv2.imread(input_data)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.filter2D(gray, -1, self.kernel)
            result = cv2.subtract(gray, blurred)
            result = cv2.add(result, 0.5 * 255)
            result = np.clip(result, 0, 255)

        else:
            raise ValueError("Unknown edge detection method.")

        return result


def _load_image(image):
    """Load image from either a file path or directly from a numpy array."""
    if isinstance(image, str):
        return image

    elif isinstance(image, np.ndarray):
        with tempfile.NamedTemporaryFile(suffix = ".png", delete = False) as temp_file:
            path = temp_file.name
            img = Image.fromarray(image)
            img.save(path)

        return path

    else:
        raise ValueError("Invalid input. Provide either a file path or a numpy array.")


def _create_gaussian_kernel(size, sigma):
    """
    Create a Gaussian kernel.

    """
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) *
                     np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size))

    return kernel / np.sum(kernel)
