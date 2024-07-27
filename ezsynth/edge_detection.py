import cv2
import numpy as np
import torch

from phycv import PAGE_GPU, PST_GPU

from .aux_classes import EdgeConfig
from .aux_utils import replace_zeros_tensor


class EdgeDetector:
    def __init__(self, method="PAGE"):
        """
        Initialize the edge detector.

        :param method: Edge detection method. Choose from 'PST', 'Classic', or 'PAGE'.
        :PST: Phase Stretch Transform (PST) edge detector. - Good overall structure, 
        but not very detailed.
        :Classic: Classic edge detector. - A good balance between structure and detail.
        :PAGE: Phase and Gradient Estimation (PAGE) edge detector. - 
        Great detail, great structure, but slow.
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
        self.pad_size = 16

    @staticmethod
    def create_gaussian_kernel(size, sigma):
        x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
        g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
        return g / g.sum()

    def pad_image(self, img):
        return cv2.copyMakeBorder(
            img,
            self.pad_size,
            self.pad_size,
            self.pad_size,
            self.pad_size,
            cv2.BORDER_REFLECT,
        )

    def unpad_image(self, img):
        return img[self.pad_size : -self.pad_size, self.pad_size : -self.pad_size]

    def classic_preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.filter2D(gray, -1, self.kernel)
        edge_map = cv2.subtract(gray, blurred)
        edge_map = np.clip(edge_map + 128, 0, 255)
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

        padded_img = self.pad_image(input_img)

        self.pst_gpu.h = padded_img.shape[0]
        self.pst_gpu.w = padded_img.shape[1]

        self.pst_gpu.img = torch.from_numpy(padded_img).to(self.pst_gpu.device)
        # If input has too many zeros the model returns NaNs for some reason
        self.pst_gpu.img = replace_zeros_tensor(self.pst_gpu.img, 1)
        
        self.pst_gpu.init_kernel(S, W)
        self.pst_gpu.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)

        edge_map = self.pst_gpu.pst_output.cpu().numpy()
        edge_map = self.unpad_image(edge_map)

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
        padded_img = self.pad_image(input_img)

        self.page_gpu.h = padded_img.shape[0]
        self.page_gpu.w = padded_img.shape[1]

        self.page_gpu.img = torch.from_numpy(padded_img).to(self.page_gpu.device)
        # If input has too many zeros the model returns NaNs for some reason
        self.page_gpu.img = replace_zeros_tensor(self.page_gpu.img, 1)
        
        self.page_gpu.init_kernel(mu_1, mu_2, sigma_1, sigma_2, S1, S2)
        self.page_gpu.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)
        self.page_gpu.create_page_edge()

        edge_map = self.page_gpu.page_edge.cpu().numpy()
        edge_map = self.unpad_image(edge_map)
        return edge_map

    def compute_edge(self, input_data: np.ndarray):
        edge_map = None
        if self.method == "PST":
            edge_map = self.pst_run(input_data, **EdgeConfig.get_pst_default())
            edge_map = self.pst_page_postprocess(edge_map)
            return edge_map

        if self.method == "Classic":
            edge_map = self.classic_preprocess(input_data)
            return edge_map

        if self.method == "PAGE":
            edge_map = self.page_run(input_data, **EdgeConfig.get_page_default())
            edge_map = self.pst_page_postprocess(edge_map)
            return edge_map
        return edge_map

