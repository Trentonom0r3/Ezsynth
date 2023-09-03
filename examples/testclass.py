from matplotlib import pyplot as plt
import torch
from utils.optical_flow import OpticalFlowProcessor
import os
import re
import numpy as np
import cv2

class Test:
    
    def __init__(self, imgsequence, flow_method='RAFT'):
        
        self.DEVICE = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') # Device to run the model on (CPU or GPU)
        
        self.flow_method = flow_method  # Optical flow method to use
        assert self.flow_method in ['RAFT', 'DeepFlow'], "Invalid flow method."
        
        if os.path.isdir(imgsequence) and imgsequence is not None:  # If it's a folder, list all image files inside it

            filenames = os.listdir(imgsequence)

            filenames.sort(
                key=lambda x: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', x)])

            self.imgsequence = [os.path.join(imgsequence, fname)
                                for fname in filenames
                                if os.path.isfile(os.path.join(imgsequence, fname)) and
                                fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            self.optical_flow_sequence = []
            
    def compute_optical_flow(self, backward=False):
        optical_flow = []
        self.flow = OpticalFlowProcessor(
            model_name='raft-sintel.pth', method=self.flow_method)

        image_batches = [(self.imgsequence[i+1], self.imgsequence[i]) if backward else (self.imgsequence[i], self.imgsequence[i+1])
                        for i in range(len(self.imgsequence)-1)]

        flow_results = self.flow.compute_optical_flow_parallel(
            image_batches, method=self.flow_method)

        optical_flow.extend(flow_results)
        return optical_flow
    
    def check_flow_consistency(self, forward_flow, backward_flow, threshold=0.1):
        # Warp the backward flow using the forward flow
        warped_backward_flow = self.warp(forward_flow, backward_flow)
        
        # Warp the forward flow using the backward flow
        warped_forward_flow = self.warp(backward_flow, forward_flow)
        
        # Compute the flow difference
        forward_diff = np.linalg.norm(forward_flow - warped_forward_flow, axis=-1)
        backward_diff = np.linalg.norm(backward_flow - warped_backward_flow, axis=-1)
        
        # Check for inconsistencies (you can adjust the threshold as needed)
        occlusions = np.logical_or(forward_diff > threshold, backward_diff > threshold)
        
        return occlusions


    def temporal_smoothing(self, window_size=3):
        # Apply a moving average filter to the optical flow sequence
        smoothed_flow_sequence = []
        for i in range(len(self.optical_flow_sequence)):
            start = max(0, i - window_size // 2)
            end = min(len(self.optical_flow_sequence), i + window_size // 2 + 1)
            average_flow = np.mean(self.optical_flow_sequence[start:end], axis=0)
            smoothed_flow_sequence.append(average_flow)

        self.optical_flow_sequence = smoothed_flow_sequence
        
    def visualize_occlusions(self, occlusion_mask, frame_index):
        # Get the original image at the given frame index
        original_image = cv2.imread(self.imgsequence[frame_index])
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Ensure the occlusion_mask has the same shape as the original image
        if occlusion_mask.shape != original_image.shape[:2]:
            print(occlusion_mask.shape, original_image.shape)
            raise ValueError("Occlusion mask shape does not match the image shape.")

        # Create a color overlay for occlusions (red color in this case)
        occlusion_overlay = np.zeros_like(original_image)
        occlusion_overlay[occlusion_mask] = [255, 0, 0]

        # Blend the original image with the occlusion overlay
        blended_image = cv2.addWeighted(original_image, 0.7, occlusion_overlay, 0.3, 0)

        # Display the blended image
        plt.imshow(blended_image)
        plt.axis('off')
        plt.show()

        
    def warp(self, flow, image):
        h, w = flow.shape[:2]
        flow_map = np.indices((h, w)).transpose(1, 2, 0).astype(np.float32) + flow
        warped_image = cv2.remap(image, flow_map, None, interpolation=cv2.INTER_LINEAR)
        return warped_image

