# import torch
from PIL import Image
import numpy as np
import os
from phycv import PST_GPU, PAGE_GPU
import cv2

import torch


class EdgeDetector:
    def __init__(self, method="PAGE"):
        """
        Initialize the edge detector.

        :param method: Edge detection method. Choose from 'PST', 'Classic', or 'PAGE'.
        :PST: Phase Stretch Transform (PST) edge detector. - Good overall structure, but not very detailed.
        :Classic: Classic edge detector. - A good balance between structure and detail.
        :PAGE: Phase and Gradient Estimation (PAGE) edge detector. - Great detail, great structure, but slow.
        :return: None.
        """
        self.method = method
        self.device = "cuda"
        if method == "PST":
            self.pst_gpu = PST_GPU(device=self.device)
        elif method == "PAGE":
            self.page_gpu = PAGE_GPU(direction_bins=10, device=self.device)
        elif method == "Classic":
            size, sigma = 5, 6.0
            self.kernel = self.create_gaussian_kernel(size, sigma)

    @staticmethod
    def save_result(output_dir, base_file_name, result_array):
        """
        Save the numpy array result to the specified directory and return the file path.
        """
        os.makedirs(output_dir, exist_ok=True)

        edge_result = Image.fromarray((result_array * 255).astype(np.uint8))

        output_file_path = os.path.join(output_dir, base_file_name)

        edge_result.save(output_file_path)

        return output_file_path

    @staticmethod
    def create_gaussian_kernel(size, sigma):
        """
        Create a Gaussian kernel.

        """
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma**2))
            * np.exp(
                -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2)
                / (2 * sigma**2)
            ),
            (size, size),
        )

        return kernel / np.sum(kernel)

    def classic_preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.filter2D(gray, -1, self.kernel)
        edge_map = cv2.subtract(gray, blurred)
        edge_map = cv2.add(edge_map, 0.5 * 255)
        edge_map = np.clip(edge_map, 0, 255)
        return edge_map.astype(np.uint8)

    def pst_page_postprocess(self, edge_map: np.ndarray):
        edge_map = cv2.GaussianBlur(edge_map, (5, 5), 3)
        edge_map = edge_map * 255
        return edge_map.astype(np.uint8)

    def pst_run(
        self,
        input_data: np.ndarray,
        S,
        W,
        sigma_LPF,
        thresh_min,
        thresh_max,
        morph_flag,
    ):
        input_img = cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY)
        self.pst_gpu.h = input_img.shape[0]
        self.pst_gpu.w = input_img.shape[1]
        self.pst_gpu.img = torch.from_numpy(input_img).to(self.pst_gpu.device)
        self.pst_gpu.init_kernel(S, W)
        self.pst_gpu.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)
        edge_map = self.pst_gpu.pst_output.cpu().numpy()
        return edge_map

    def page_run(
        self,
        input_data: np.ndarray,
        mu_1,
        mu_2,
        sigma_1,
        sigma_2,
        S1,
        S2,
        sigma_LPF,
        thresh_min,
        thresh_max,
        morph_flag,
    ):
        input_img = cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY)
        self.page_gpu.h = input_img.shape[0]
        self.page_gpu.w = input_img.shape[1]
        self.page_gpu.img = torch.from_numpy(input_img).to(self.page_gpu.device)
        self.page_gpu.init_kernel(mu_1, mu_2, sigma_1, sigma_2, S1, S2)
        self.page_gpu.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)
        self.page_gpu.create_page_edge()
        edge_map = self.page_gpu.page_edge.cpu().numpy()
        return edge_map

    def compute_edge(self, input_data: np.ndarray):
        """
        Compute the edge map.

        :param input_data: Either a file path or a numpy array.

        :return: Edge map as a numpy array.
        """
        if self.method == "PST":
            edge_map = self.pst_run(
                input_data,
                S=0.3,
                W=15,
                sigma_LPF=0.15,
                thresh_min=0.05,
                thresh_max=0.9,
                morph_flag=1,
            )
            edge_map = self.pst_page_postprocess(edge_map)
            return edge_map

        if self.method == "Classic":
            edge_map = self.classic_preprocess(input_data)
            return edge_map

        if self.method == "PAGE":
            edge_map = self.page_run(
                input_data,
                mu_1=0,
                mu_2=0.35,
                sigma_1=0.05,
                sigma_2=0.8,
                S1=0.8,
                S2=0.8,
                sigma_LPF=0.1,
                thresh_min=0.0,
                thresh_max=0.9,
                morph_flag=1,
            )
            edge_map = self.pst_page_postprocess(edge_map)
            return edge_map

        return edge_map
