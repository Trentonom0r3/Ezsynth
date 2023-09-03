import cv2
import numpy as np
import torch
from utils.edge_detection import EdgeDetector
from utils.optical_flow import OpticalFlowProcessor

class Guides:
    def __init__(self):
        pass
    
    def compute_edge_guides(self, imgsequence, edge_method):
        """
        Compute edge guides for the image sequence.

        Parameters
        ----------
        imgsequence : List of images as file paths or numpy arrays.

        Returns
        -------
        edge_guides : List of edge guides as numpy arrays.

        """
        edge_guides = []

        edge_detector = EdgeDetector(method=edge_method)

        print("[INFO] Computing Edge Guides...")

        for img in imgsequence:  # compute edge guides for each image in imgsequence

            edge_guides.append(edge_detector.compute_edge(img))
            
        print("[INFO] Edge Guides Computed.")
        
        return edge_guides
    
    def compute_optical_flow(self, imgsequence, flow_method):
        """
        Compute optical flow for the image sequence.

        Parameters
        ----------
        imgsequence : List of images as file paths or numpy arrays.

        Returns
        -------
        optical_flow : List of optical flow results as numpy arrays.
        """
        optical_flow = []

        flow = OpticalFlowProcessor(
            model_name='raft-sintel.pth', method=flow_method)

        image_batches = [(imgsequence[i], imgsequence[i+1])  # Create batches of image pairs
                         for i in range(len(imgsequence)-1)]

        # Compute optical flow in parallel for the entire batch
        flow_results = flow.compute_optical_flow(
            image_batches, method=flow_method)

        optical_flow.extend(flow_results)

        return optical_flow
    
    def create_g_pos(self, optical_flow, imgsequence, reverse = False):
        """
        Create g_pos files for the image sequence.

        Parameters
        ----------
        optical_flow : List of optical flow results as numpy arrays.
        imgsequence : List of images as file paths or numpy arrays.

        Returns
        -------
        g_pos_files : List of g_pos files as numpy arrays.
        """
        g_pos_files = []
        ORIGINAL_SIZE = cv2.imread(imgsequence[0]).shape[1::-1]
        if reverse == True:
            optical_flow = optical_flow
        else:
            optical_flow = optical_flow[::-1]
        flow = OpticalFlowProcessor(
            model_name='raft-sintel.pth', method="RAFT")
        
        for i in range(len(optical_flow)):
            
            g_pos = flow.create_g_pos_from_flow(
                optical_flow[i], ORIGINAL_SIZE)
            g_pos_files.append(g_pos)
            
        if reverse == True:
            g_pos_files = g_pos_files
        else:
            g_pos_files = g_pos_files[::-1]
        
        return g_pos_files
    
    def warp_masks(self, optical_flow, err_masks):
        """
        Warp error masks using optical flow.
        
        Parameters
        ----------
        optical_flow : List of optical flow results as numpy arrays.
        err_masks : List of error masks as numpy arrays.
        
        Returns
        -------
        warped_masks : List of warped error masks as numpy arrays.
        """
        warped_masks = []
        ORIGINAL_SIZE = err_masks[0].shape[1::-1]
        self.flow = OpticalFlowProcessor(
            model_name='raft-sintel.pth', method="RAFT")
        DEVICE = 'cuda'
        for i in range(len(err_masks)-1):
                
                #check if mask is 3 channel
                if len(err_masks[i].shape) == 2:
                    err_masks[i] = np.repeat(err_masks[i][:, :, np.newaxis], 3, axis=2)
                              
                mask = torch.from_numpy(
                    err_masks[i]).permute(2, 0, 1).float()

                mask = mask.unsqueeze(0).to(DEVICE)

                flow = torch.from_numpy(
                    optical_flow[i-1]).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

                flow *= -1

                warped_mask = self.flow.warp(mask, flow)

                warped_mask = warped_mask.squeeze(
                    0).permute(1, 2, 0).cpu().detach().numpy()

               
                warped_mask_np = np.clip(warped_mask, 0, 1)

                warped_mask_np = (
                    warped_mask_np * 255).astype(np.uint8)

                # compare warped_mask_np with err_masks[i+1]
                # if pixel is white in i+1, but black after warping, change it to white
                err_masks[i+1] = np.repeat(err_masks[i+1][:, :, np.newaxis], 3, axis=2)
                mask_np = np.where(
                    (warped_mask_np == 0) & (err_masks[i+1] == 255), 255, warped_mask_np)
                
                warped_mask_np = warped_mask_np + mask_np
                warped_mask_np = np.clip(warped_mask_np, 0, 255)
                
                
                warped_masks.append(warped_mask_np)
        return warped_masks