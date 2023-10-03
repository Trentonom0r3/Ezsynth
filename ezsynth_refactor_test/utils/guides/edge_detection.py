import torch
from PIL import Image
import numpy as np
import os
from phycv import PST_GPU, PAGE_GPU
import cv2
import tempfile


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
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        if method == "PST":
            self.pst_gpu = PST_GPU(device=self.device)
        elif method == "PAGE":
            self.page_gpu = PAGE_GPU(direction_bins=10, device=self.device)
        elif method == "Classic":
            size, sigma = 5, 6.0
            self.kernel = self.create_gaussian_kernel(size, sigma)
        
    
    @staticmethod
    def load_image(input_data):
        """Load image from either a file path or directly from a numpy array."""
        if isinstance(input_data, str):  # If it's a file path
            return input_data
        # If it's a numpy array, save it as a temporary file
        elif isinstance(input_data, np.ndarray):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_file_path = temp_file.name
                img = Image.fromarray(input_data)
                img.save(temp_file_path)

            return temp_file_path
        else:
            raise ValueError(
                "Invalid input. Provide either a file path or a numpy array.")

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
            lambda x, y: (1/(2*np.pi*sigma**2)) *
            np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2) / (2*sigma**2)),
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
                pst_gpu = PST_GPU(device=self.device)
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
                page_gpu = PAGE_GPU(direction_bins=10, device=self.device)
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
