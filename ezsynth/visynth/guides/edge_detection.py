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
            size, sigma = 5, 6.0
            self.kernel = self.create_gaussian_kernel(size, sigma)
        else:
            raise ValueError("Unknown edge detection method.")

    @staticmethod
    def load_image(image):
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

    @staticmethod
    def save_result(output_dir, base_file_name, result_array):
        """
        Save the numpy array result to the specified directory and return the file path.
        """
        os.makedirs(output_dir, exist_ok = True)

        result = Image.fromarray((result_array * 255).astype(np.uint8))

        path = os.path.join(output_dir, base_file_name)

        result.save(path)

        return path

    @staticmethod
    def create_gaussian_kernel(size, sigma):
        """
        Create a Gaussian kernel.

        """
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) *
                         np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
            (size, size))

        return kernel / np.sum(kernel)

    def compute_edge(self, input_data):
        """
        Compute the edge map.

        :param input_data: Either a file path or a numpy array.

        :return: Edge map as a numpy array.
        """
        method = self.method
        if method == 'Classic':
            pass
        elif method == 'PST':
            input_data_path = self.load_image(input_data)
        elif method == 'PAGE':
            input_data_path = self.load_image(input_data)
        else:
            raise ValueError(
                "Invalid method. Choose from 'PST', 'Classic', or 'PAGE'.")

        try:
            if method == "PST":
                # pst_gpu = PST_GPU(device=self.device)
                S, W, sigma_LPF, thresh_min, thresh_max, morph_flag = 0.3, 15, 0.15, 0.05, 0.9, 1
                edge_map = self.pst_gpu.run(
                    input_data_path, S, W, sigma_LPF, thresh_min, thresh_max, morph_flag)
                edge_map = edge_map.cpu().numpy()
                edge_map = cv2.GaussianBlur(edge_map, (5, 5), 3)
                edge_map = (edge_map * 255).astype(np.uint8)

            elif self.method == "Classic":
                if isinstance(input_data, np.ndarray):
                    img = input_data
                elif isinstance(input_data, str):
                    img = cv2.imread(input_data)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blurred = cv2.filter2D(gray, -1, self.kernel)
                edge_map = cv2.subtract(gray, blurred)
                edge_map = cv2.add(edge_map, 0.5 * 255)
                edge_map = np.clip(edge_map, 0, 255)

            elif method == "PAGE":
                # page_gpu = PAGE_GPU(direction_bins=10, device=self.device)
                mu_1, mu_2, sigma_1, sigma_2, S1, S2, sigma_LPF, thresh_min, thresh_max, morph_flag = 0, 0.35, 0.05, 0.8, 0.8, 0.8, 0.1, 0.0, 0.9, 1
                edge_map = self.page_gpu.run(
                    input_data_path, mu_1, mu_2, sigma_1, sigma_2, S1, S2, sigma_LPF, thresh_min, thresh_max, morph_flag)
                edge_map = edge_map.cpu().numpy()
                edge_map = cv2.GaussianBlur(edge_map, (5, 5), 3)
                edge_map = (edge_map * 255).astype(np.uint8)


            else:
                raise ValueError(
                    "Invalid method. Choose from 'PST', 'Guide', or 'PAGE'.")
        finally:
            # If the input_data was a numpy array and the method is not 'Classic', delete the temporary file
            if method != 'Classic' and isinstance(input_data, np.ndarray):
                os.remove(input_data_path)

        return edge_map
