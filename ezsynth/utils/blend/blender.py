from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..flow_utils.warp import Warp
from .histogram_blend import HistogramBlender
from .reconstruction import reconstructor


class Blend:
    def __init__(
        self,
        style_fwd: list[np.ndarray],
        style_bwd: list[np.ndarray],
        err_fwd: list[np.ndarray],
        err_bwd: list[np.ndarray],
        flow_fwd: list[np.ndarray],
    ):
        self.style_fwd = style_fwd
        self.style_bwd = style_bwd
        self.err_fwd = err_fwd
        self.err_bwd = err_bwd
        self.flow = flow_fwd
        self.err_masks = None
        self.blends = None

    def _create_final_err_masks(self):
        err_masks = self._create_selection_mask(self.err_fwd, self.err_bwd)
        print(f"{len(err_masks)=}")
        print(f"{err_masks[0].shape=}")
        print(f"{type(err_masks[0])=}")
        # Check if err_masks is empty
        if not err_masks:
            print("Error: err_masks is empty.")
            return []
        # use err_masks with flow to create final err_masks
        self.prev_mask = None

        ORIGINAL_SIZE = self.style_fwd[0].shape
        print(f"Original size: {ORIGINAL_SIZE}")
        warped_masks = [None] * len(
            err_masks
        )  # Initialize with None to maintain list size
        warp = Warp(self.style_fwd[0])
        for i in range(len(err_masks) - 1):
            if self.prev_mask is None:
                self.prev_mask = np.zeros_like(err_masks[0])
            warped_mask = warp.run_warping(
                err_masks[i], self.flow[i] if i == 0 else self.flow[i - 1]
            )

            z_hat = warped_mask.copy()
            print(f"Shape of z_hat: {z_hat.shape}")
            print(f"Shape of self.prev_mask: {self.prev_mask.shape}")
            # If the shapes are not compatible, we can adjust the shape of self.prev_mask
            if self.prev_mask.shape != z_hat.shape:
                self.prev_mask = np.repeat(
                    self.prev_mask[:, :, np.newaxis], z_hat.shape[2], axis=2
                )

            z_hat = np.where((self.prev_mask > 1) & (z_hat == 0), 1, z_hat)

            self.prev_mask = z_hat.copy()
            warped_masks[i] = z_hat.copy()

        # Safely assign the last element
        if len(err_masks) > 0:
            warped_masks[-1] = err_masks[-1]

        return warped_masks

    def _create_selection_mask(
        self, err_forward_lst: list[np.ndarray], err_backward_lst: list[np.ndarray]
    ) -> list[np.ndarray]:
        # Convert lists to numpy arrays
        err_forward = np.array(err_forward_lst)
        err_backward = np.array(err_backward_lst)

        # Check that shapes match
        if err_forward.shape != err_backward.shape:
            print(f"Shape mismatch: {err_forward.shape=} vs {err_backward.shape=}")
            return []  # Return an empty list if shapes don't match

        # Create a binary mask where the forward error metric is less than the backward error metric
        selection_masks = np.where(err_forward < err_backward, 0, 1).astype(np.uint8)

        # Convert numpy array back to list
        selection_masks_lst = [
            selection_masks[i] for i in range(selection_masks.shape[0])
        ]

        return selection_masks_lst

    def _hist_blend(self):
        hist_blends = []
        with ThreadPoolExecutor() as executor:
            for i in range(len(self.err_masks)):
                future = executor.submit(
                    HistogramBlender().blend,
                    self.style_fwd[i],
                    self.style_bwd[i],
                    self.err_masks[i],
                )
                hist_blends.append(future.result())
        print(len(hist_blends))
        return hist_blends

    def _reconstruct(self, hist_blends):
        blends = reconstructor(
            hist_blends, self.style_fwd, self.style_bwd, self.err_masks
        )
        final_blends = blends()
        final_blends = [blend for blend in final_blends if blend is not None]
        return final_blends

    def __call__(self):
        self.err_masks = self._create_final_err_masks()
        hist_blends = self._hist_blend()
        self.blends = self._reconstruct(hist_blends)
        return self.blends
