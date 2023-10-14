from concurrent.futures import ThreadPoolExecutor
import numpy as np
from utils.blend.histogram_blend import HistogramBlender
from utils.blend.reconstruction import reconstructor
from utils.flow_utils.warp import Warp

class Blend():
    def __init__(self, style_fwd, style_bwd, err_fwd, err_bwd, flow_fwd):
        self.style_fwd = style_fwd
        self.style_bwd = style_bwd
        self.err_fwd = err_fwd
        self.err_bwd = err_bwd
        self.flow = flow_fwd
        self.err_masks = None
        self.blends = None
        pass
    
    def _create_final_err_masks(self):
        err_masks = self._create_selection_mask(self.err_fwd, self.err_bwd)


        # use err_masks with flow to create final err_masks
        self.prev_mask = None
        ORIGINAL_SIZE = self.style_fwd[0].shape
        warped_masks = [None] * len(err_masks)  # Initialize with None to maintain list size

        warp = Warp(self.style_fwd[0])
        for i in range(len(err_masks) - 1):

            if self.prev_mask is None:
                self.prev_mask = np.zeros_like(err_masks[0])
            warped_mask = warp.run_warping(err_masks[i], self.flow[i] if i == 0 else self.flow[i - 1])

            z_hat = warped_mask.copy()
            
            # If the shapes are not compatible, we can adjust the shape of self.prev_mask
            if self.prev_mask.shape != z_hat.shape:
                self.prev_mask = np.repeat(self.prev_mask[:, :, np.newaxis], z_hat.shape[2], axis=2)

            z_hat = np.where((self.prev_mask > 1) & (z_hat == 0), 1, z_hat)

            
            self.prev_mask = z_hat.copy()
            warped_masks[i] = z_hat.copy()
            
        warped_masks[-1] = err_masks[-1]

        return warped_masks
    
    def _create_selection_mask(self, err_forward, err_backward):
        selection_masks = []

        for forward_list, backward_list in zip(err_forward, err_backward):
            
            # Make sure both are lists
            if not isinstance(forward_list, list) or not isinstance(backward_list, list):
                print(f"Forward and backward should be lists of numpy arrays.")
                continue


            # Now we loop through the individual arrays in each list and make selections
            for forward, backward in zip(forward_list, backward_list):
                
                # Convert to NumPy array if they are not
                forward = np.array(forward) if not isinstance(forward, np.ndarray) else forward
                backward = np.array(backward) if not isinstance(backward, np.ndarray) else backward
                

                # Check that shapes match
                if forward.shape != backward.shape:
                    print(f"Shape mismatch: {forward.shape} vs {backward.shape}")
                    continue  # Skip this iteration
                    
                # Create a binary mask where the forward error metric is less than the backward error metric
                selection_mask = np.where(forward < backward, 0, 1)
                selection_mask = selection_mask.astype(np.uint8)

                # Add to the list of masks
                selection_masks.append(selection_mask)

        return selection_masks

        
    def _hist_blend(self):
        hist_blends = []
        with ThreadPoolExecutor() as executor:
            for i in range(len(self.err_masks)):
                future = executor.submit(HistogramBlender().blend, self.style_fwd[i], self.style_bwd[i], self.err_masks[i])
                hist_blends.append(future.result())
                
        return hist_blends
    
    def _reconstruct(self, hist_blends):
        blends = reconstructor(hist_blends, self.style_fwd, self.style_bwd, self.err_masks)
        final_blends = blends()
        return final_blends
    
    def __call__(self):
        self.err_masks = self._create_final_err_masks()
        hist_blends = self._hist_blend()
        self.blends = self._reconstruct(hist_blends)
        return self.blends
    
    
