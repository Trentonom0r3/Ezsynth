from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .histogram_blend import HistogramBlender
from .reconstruction import reconstructor
from ..flow_utils.warp import Warp


class Blend:
    def __init__(self, style_fwd, style_bwd, err_fwd, err_bwd, flow_fwd):
        self.style_fwd = style_fwd
        self.style_bwd = style_bwd
        self.err_fwd = err_fwd
        self.err_bwd = err_bwd
        self.flow = flow_fwd
        self.err_masks = None
        self.blends = None

    def __call__(self):
        self.err_masks = self._create_final_err_masks()
        hist_blends = self._hist_blend()
        self.blends = self._reconstruct(hist_blends)
        return self.blends

    def _create_final_err_masks(self):
        err_masks = self._create_selection_mask(self.err_fwd, self.err_bwd)
        print(f"Length of err_masks: {len(err_masks)}")
        print(f"Shape of err_masks[0]: {err_masks[0].shape}")
        print(f"Type of err_masks[0]: {type(err_masks[0])}")

        if not err_masks:
            print("Error: err_masks is empty.")
            return []

        # use err_masks with flow to create final err_masks
        self.prev_mask = None

        print(f"Original size: {self.style_fwd[0].shape}")
        warped_masks = [None] * len(err_masks)  # Initialize with None to maintain list size
        warp = Warp(self.style_fwd[0])
        for i in range(len(err_masks) - 1):

            if self.prev_mask is None:
                self.prev_mask = np.zeros_like(err_masks[0])
            warped_mask = warp.run_warping(err_masks[i], self.flow[i] if i == 0 else self.flow[i - 1])

            z_hat = warped_mask.copy()
            print(f"Shape of z_hat: {z_hat.shape}")
            print(f"Shape of self.prev_mask: {self.prev_mask.shape}")
            # If the shapes are not compatible, we can adjust the shape of self.prev_mask
            if self.prev_mask.shape != z_hat.shape:
                self.prev_mask = np.repeat(self.prev_mask[:, :, np.newaxis], z_hat.shape[2], axis = 2)

            z_hat = np.where((self.prev_mask > 1) & (z_hat == 0), 1, z_hat)

            self.prev_mask = z_hat.copy()
            warped_masks[i] = z_hat.copy()

        # Safely assign the last element
        if len(err_masks) > 0:
            warped_masks[-1] = err_masks[-1]

        return warped_masks

    def _create_selection_mask(self, err_forward, err_backward):
        selection_masks = []

        for i in range(len(err_forward)):

            # Check that shapes match
            if err_forward[i].shape != err_backward[i].shape:
                print(f"Shape mismatch: {err_forward.shape} vs {err_backward.shape}")
                continue  # Skip this iteration

            # Create a binary mask where the forward error metric is less than the backward error metric
            selection_mask = np.where(err_forward[i] < err_backward[i], 0, 1)
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
        print(len(hist_blends))
        return hist_blends

    def _reconstruct(self, hist_blends):
        blends = reconstructor(hist_blends, self.style_fwd, self.style_bwd, self.err_masks)
        final_blends = blends()
        final_blends = [blend for blend in final_blends if blend is not None]
        return final_blends
