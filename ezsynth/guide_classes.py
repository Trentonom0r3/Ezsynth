import cv2
import numpy as np
import torch
from .edge_detection import EdgeDetector
from .optical_flow import OpticalFlowProcessor

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


        for img in imgsequence:  # compute edge guides for each image in imgsequence

            edge_guides.append(edge_detector.compute_edge(img))

        return edge_guides
    
    def compute_optical_flow(self, imgsequence, flow_method, flow_model):
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

        flow = OpticalFlowProcessor(method=flow_method, model_name_descriptor = flow_model)

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

        ORIGINAL_SIZE = cv2.imread(imgsequence[0]).shape[1::-1]

        flow = OpticalFlowProcessor(
           model_name_descriptor ='sintel', method="RAFT")
        
        if reverse == True:
            optical_flow = [optical_flow[i]*-1 for i in range(len(optical_flow))]
            
            optical_flow = optical_flow
            g_pos = flow.create_g_pos_from_flow(
                    optical_flow, ORIGINAL_SIZE)
       
        else:
            optical_flow = optical_flow[::-1]
                          
            g_pos = flow.create_g_pos_from_flow(
                        optical_flow, ORIGINAL_SIZE)
            
            g_pos = g_pos[::-1] # do not change form this-- works okay for now. Do I need to invert flow?
    
        return g_pos
    
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
        try:
            self.prev_mask = None
            
            warped_masks = [None] * len(err_masks)  # Initialize with None to maintain list size

            ORIGINAL_SIZE = err_masks[0].shape[1::-1]
            self.flow = OpticalFlowProcessor(model_name_descriptor ='sintel', method="RAFT")
            DEVICE = 'cuda'

            for i in range(len(err_masks) - 1):

                # Maintain Previous Mask
                if self.prev_mask is None:
                    self.prev_mask = np.zeros_like(err_masks[0])

                # Existing code to warp the mask
                if len(err_masks[i].shape) == 2:
                    err_masks[i] = np.repeat(err_masks[i][:, :, np.newaxis], 3, axis=2)
                
                mask = torch.from_numpy(err_masks[i]).permute(2, 0, 1).float()
                mask = mask.unsqueeze(0).to(DEVICE)

                flow = torch.from_numpy(optical_flow[i-1]).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
                flow *= -1

                warped_mask = self.flow.warp(mask, flow)
                warped_mask = warped_mask.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                warped_mask_np = np.clip(warped_mask, 0, 1)
                warped_mask_np = (warped_mask_np * 255).astype(np.uint8)
                
                # Temporal Coherence
                # Assuming `warped_mask_np` is your Z_hat
                z_hat = warped_mask_np
                 # Ensure self.prev_mask and z_hat have the same number of channels
                if len(self.prev_mask.shape) == 2 and len(z_hat.shape) == 3:
                    self.prev_mask = np.repeat(self.prev_mask[:, :, np.newaxis], 3, axis=2)

                # Prevent Backtracking
                z_hat = np.where((self.prev_mask > 1) & (z_hat == 0), 1, z_hat)
                
                # Store the current mask for use in the next frame
                # Convert back to single-channel if it is a 3-channel image
                if len(z_hat.shape) == 3:
                    z_hat = np.mean(z_hat, axis=2).astype(np.uint8)

                # Store the current mask for use in the next frame
                self.prev_mask = z_hat

                warped_masks[i] = z_hat  # Update the corresponding position in the list
            
             # Set the last warped mask to be the same as the last original mask
            warped_masks[-1] = err_masks[-1]

            return warped_masks
        except Exception as e:
            print(e)
            return None
        