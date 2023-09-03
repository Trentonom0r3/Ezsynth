from utils.optical_flow import OpticalFlowProcessor
import numpy as np
import os
import re
import cv2
from scipy.ndimage import map_coordinates
from scipy.fftpack import fftn, ifftn

def poisson_reconstruction(divergence, boundary_image):
    # Compute the Fourier transform of the divergence
    f_div = fftn(divergence)

    # Create the Poisson kernel in the frequency domain
    y_freq = np.fft.fftfreq(divergence.shape[0])
    x_freq = np.fft.fftfreq(divergence.shape[1])
    radius = y_freq[:, None]**2 + x_freq[None, :]**2
    radius[0, 0] = 1  # Avoid division by zero
    kernel = np.fft.fft2((-4 * np.pi**2 * radius))[:,:,np.newaxis]


    # Compute the Fourier transform of the solution
    f_solution = f_div / kernel

    # Invert the Fourier transform to get the solution
    solution = np.real(ifftn(f_solution))

    # Add the boundary image
    reconstructed_image = solution + boundary_image

    # Normalize and convert to the appropriate data type
    reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

    return reconstructed_image

def compute_gradients(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return sobelx, sobely

class smooth_flow:
    """
    A Test Class for smoothing optical flow.
    """

    def __init__(self, input_seq, style_seq, flow_method="RAFT"):
        """
        Initialize the smoothing class.

        :param input_seq: List of images as file paths or numpy arrays.
        :param style_seq: List of images as file paths or numpy arrays.
        :param flow_method: Optical flow method. Choose from 'RAFT' or 'DeepFlow'.
        :return: None.
        """
        self.flow_method = flow_method
        self.flow = OpticalFlowProcessor(
            model_name='raft-sintel.pth', method=self.flow_method)

        if os.path.isdir(input_seq):  # If it's a folder, list all image files inside it

            filenames = os.listdir(input_seq)

            filenames.sort(
                key=lambda x: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', x)])

            self.input_seq = [os.path.join(input_seq, fname)
                                for fname in filenames
                                if os.path.isfile(os.path.join(input_seq, fname)) and
                                fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
        if os.path.isdir(style_seq):  # If it's a folder, list all image files inside it

            filenames = os.listdir(style_seq)

            filenames.sort(
                key=lambda x: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', x)])

            self.style_seq = [os.path.join(style_seq, fname)
                                for fname in filenames
                                if os.path.isfile(os.path.join(style_seq, fname)) and
                                fname.lower().endswith(('.png', '.jpg', '.jpeg'))]


        self.input_flow, self.input_temporal_smoothness = self._compute_optical_flow(self.input_seq)
        self.style_flow, self.style_temporal_smoothness = self._compute_optical_flow(self.style_seq)

         # Calculate mapping ratio
        self.map_ratio = self.find_mapping_ratio(self.style_flow, self.input_flow, self.input_temporal_smoothness)
        
    def _compute_optical_flow(self, imgsequence):
        # Constants for the optimization
        alpha = 0.1  # Weight for the regularization term

        flow_results = []
        temporal_smoothness_results = []

        # Prepare batches of image pairs
        image_batches_forward = [(imgsequence[i], imgsequence[i + 1]) for i in range(len(imgsequence) - 1)]
        image_batches_backward = [(imgsequence[i], imgsequence[i - 1]) for i in range(1, len(imgsequence))]

        # Compute optical flow in parallel for the entire batch (forward and backward)
        forward_flow_results, _ = self.flow.compute_optical_flow_parallel(image_batches_forward, method=self.flow_method)
        backward_flow_results, _ = self.flow.compute_optical_flow_parallel(image_batches_backward, method=self.flow_method)

        for i in range(1, len(imgsequence) - 1):
            backward_flow = backward_flow_results[i - 1]
            forward_flow = forward_flow_results[i]

            # Compute temporal smoothness
            temporal_smoothness = np.abs(forward_flow - backward_flow)

            # Apply Anisotropic Diffusion to the temporal smoothness
            temporal_smoothness = self._apply_anisotropic_diffusion(temporal_smoothness)

            # Store the results
            flow_results.append((backward_flow, forward_flow))
            temporal_smoothness_results.append(temporal_smoothness)

        return flow_results, temporal_smoothness_results



    def _apply_anisotropic_diffusion(self, temporal_smoothness, iterations=10, kappa=20, lambda_val=0.1):
        """
        Apply Anisotropic Diffusion to the temporal smoothness.

        :param temporal_smoothness: Temporal smoothness as numpy array.
        :param iterations: Number of diffusion iterations.
        :param kappa: Edge threshold parameter.
        :param lambda_val: Diffusion speed parameter.

        :return: Diffused temporal smoothness as numpy array.
        """
        for _ in range(iterations):
            # Compute the gradient along the temporal dimension
            gradient_temporal = np.gradient(temporal_smoothness, axis=0)

            # Compute the diffusion coefficient based on local characteristics
            c = np.exp(-np.abs(gradient_temporal) / kappa)

            # Apply diffusion
            temporal_smoothness += lambda_val * c * gradient_temporal

        return temporal_smoothness

    def _objective_function(self, map_ratio, style_flow, input_flow, temporal_smoothness, alpha=0.1):
        error = 0
        for (style_backward, style_forward), (input_backward, input_forward), smoothness in zip(style_flow, input_flow, temporal_smoothness):
            data_term_backward = np.sum((style_backward - map_ratio * input_backward) ** 2)
            data_term_forward = np.sum((style_forward - map_ratio * input_forward) ** 2)
            data_term = data_term_backward + data_term_forward
            regularization_term = np.sum(smoothness ** 2)
            error += data_term + alpha * regularization_term
        return error


    def _derivative_objective_function(self, map_ratio, style_flow, input_flow, temporal_smoothness, epsilon=1e-5, alpha=0.1):
        e1 = self._objective_function(map_ratio + epsilon, style_flow, input_flow, temporal_smoothness, alpha)
        e2 = self._objective_function(map_ratio - epsilon, style_flow, input_flow, temporal_smoothness, alpha)
        derivative = (e1 - e2) / (2 * epsilon)
        return derivative

    def find_mapping_ratio(self, style_flow, input_flow, temporal_smoothness, initial_guess=0.42, tolerance=0.15, max_iterations=100):
        """
        Find the mapping ratio using the Newton-Raphson method.

        :param style_flow: Optical flow of the stylized sequence.
        :param input_flow: Optical flow of the original sequence.
        :param temporal_smoothness: Smoothed temporal smoothness.
        :param initial_guess: Initial guess for the mapping ratio.
        :param tolerance: Tolerance level for convergence.
        :param max_iterations: Maximum number of iterations.
        :return: Optimal mapping ratio as a float.
        """
        map_ratio = initial_guess
        for iteration in range(max_iterations):
            error = self._objective_function(map_ratio, style_flow, input_flow, temporal_smoothness)
            derivative = self._derivative_objective_function(map_ratio, style_flow, input_flow, temporal_smoothness)
            
            # Check if the derivative is too close to zero
            if abs(derivative) < 1e-9:
                print("Derivative too close to zero. Stopping iteration.")
                break

            # Update the mapping ratio using the Newton-Raphson formula
            map_ratio = map_ratio - error / derivative

            # Check for convergence
            if abs(error) < tolerance:
                print(f"Converged in {iteration} iterations.")
                break

        return map_ratio

    def apply_temporal_coherence(self):
        """
        Apply temporal coherence to the style sequence by using the computed mapping ratio.

        :return: The smoothed and temporally coherent style sequence.
        """
        # Compute the temporal smoothness
        temporal_smoothness_results = [self._apply_anisotropic_diffusion(smoothness) for smoothness in self._compute_optical_flow(self.input_seq)[1]]

        # Apply the mapping ratio selectively based on thresholds
        smoothed_style_flow = []
        for (style_backward, style_forward), (input_backward, input_forward) in zip(self.style_flow, self.input_flow):
            difference_backward = np.abs(style_backward - input_backward)
            difference_forward = np.abs(style_forward - input_forward)
            mask_backward = difference_backward > self.compute_thresholds(self.compute_stdev())
            mask_forward = difference_forward > self.compute_thresholds(self.compute_stdev())

            # Applying mapping ratio where mask is True, else keep original value
            smoothed_flow_backward = np.where(mask_backward, style_backward * self.map_ratio, style_backward)
            smoothed_flow_forward = np.where(mask_forward, style_forward * self.map_ratio, style_forward)
            smoothed_style_flow.append((smoothed_flow_backward, smoothed_flow_forward))

        # Warp the stylized sequence using the smoothed optical flow
        smoothed_stylized_sequence = []
        for i, (backward_flow, forward_flow) in enumerate(smoothed_style_flow):
            img = cv2.imread(self.style_seq[i])
            # Use either forward_flow or backward_flow depending on your requirements
            warped_img = self.warp_image(img, forward_flow)  # or backward_flow
            smoothed_stylized_sequence.append(warped_img)

        return smoothed_stylized_sequence

    def compute_thresholds(self, stdev):
        """
        Compute thresholds based on the standard deviation.

        :param stdev: Standard deviation of the differences between the style and input flow.
        :return: Threshold value.
        """
        # Example: Set the threshold as some multiple of the standard deviation
        # You can modify this logic as needed
        threshold_factor = 0.01
        return threshold_factor * stdev

    def compute_stdev(self):
        """
        Compute the standard deviation of the differences between the style and input flow.

        :return: Standard deviation value.
        """
        # Calculate the differences for both forward and backward flows
        differences = [np.abs(style_backward - input_backward) for (style_backward, _), (input_backward, _) in zip(self.style_flow, self.input_flow)]
        differences += [np.abs(style_forward - input_forward) for (_, style_forward), (_, input_forward) in zip(self.style_flow, self.input_flow)]
        
        # Flatten the differences and calculate the standard deviation
        stdev = np.std(np.concatenate(differences))
        return stdev
    
    def warp_gradients(self, sobelx, sobely, flow):
        warped_sobelx = self.warp_image(sobelx, flow)
        warped_sobely = self.warp_image(sobely, flow)
        return warped_sobelx, warped_sobely

    def warp_image(self, img, flow):
        """
        Warp the given image based on the provided optical flow.

        :param img: Image as a numpy array.
        :param flow: Optical flow as a numpy array.
        :return: Warped image as a numpy array.
        """
        # Resize the image to match the flow's dimensions
        resized_img = cv2.resize(img, (flow.shape[1], flow.shape[0]))

        # Compute the new coordinates using the flow
        h, w, _ = resized_img.shape
        coords = np.mgrid[0:h, 0:w].reshape(2, -1)
        coords_new = coords + flow.transpose(2, 0, 1).reshape(2, -1)

        # Map the coordinates to get the warped image
        warped_img = np.empty_like(resized_img)
        for i in range(3):  # For each color channel
            warped_img[..., i] = map_coordinates(resized_img[..., i], coords_new, order=2, mode='nearest').reshape(h, w)

        return warped_img

    def apply_guided_warping(self):
        guided_warped_sequence = []
        for i, (backward_flow, forward_flow) in enumerate(self.style_flow):
            # Read the original style image
            style_img = cv2.imread(self.style_seq[i])

            # Compute the gradients
            sobelx, sobely = compute_gradients(style_img)

            # Warp the original image
            warped_img = self.warp_image(style_img, forward_flow)

            # Warp the gradients
            warped_sobelx, warped_sobely = self.warp_gradients(sobelx, sobely, forward_flow)

            # Compute the divergence of the warped gradients
            divergence_x = cv2.Sobel(warped_sobelx, cv2.CV_64F, 1, 0, ksize=3)
            divergence_y = cv2.Sobel(warped_sobely, cv2.CV_64F, 0, 1, ksize=3)
            divergence = divergence_x + divergence_y
            
            # Reconstruct the image using Poisson reconstruction
            reconstructed_img = poisson_reconstruction(divergence, warped_img)

            # Add to the sequence
            guided_warped_sequence.append(reconstructed_img)

        return guided_warped_sequence
