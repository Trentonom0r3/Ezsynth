import time

import numpy as np
import tqdm

from ..flow_utils.warp import Warp
from .histogram_blend import hist_blender
from .reconstruction import reconstructor

try:
    from .cupy_accelerated import hist_blend_cupy

    USE_GPU = True
except ImportError as e:
    print(f"Cupy is not installed. Revert to CPU. {e}")
    USE_GPU = False


class Blend:
    def __init__(
        self,
        use_gpu=False,
        use_lsqr=True,
        use_poisson_cupy=False,
        poisson_maxiter=None,
    ):
        self.prev_mask = None

        self.use_gpu = use_gpu and USE_GPU
        self.use_lsqr = use_lsqr
        self.use_poisson_cupy = use_poisson_cupy
        self.poisson_maxiter = poisson_maxiter

    def _warping_masks(
        self,
        sample_fr: np.ndarray,
        flow_fwd: list[np.ndarray],
        err_masks: list[np.ndarray],
    ):
        # use err_masks with flow to create final err_masks
        warped_masks = []
        warp = Warp(sample_fr)

        for i in tqdm.tqdm(range(len(err_masks)), desc="Warping masks"):
            if self.prev_mask is None:
                self.prev_mask = np.zeros_like(err_masks[0])
            warped_mask = warp.run_warping(
                err_masks[i], flow_fwd[i] if i == 0 else flow_fwd[i - 1]
            )

            z_hat = warped_mask.copy()
            # If the shapes are not compatible, we can adjust the shape of self.prev_mask
            if self.prev_mask.shape != z_hat.shape:
                self.prev_mask = np.repeat(
                    self.prev_mask[:, :, np.newaxis], z_hat.shape[2], axis=2
                )

            z_hat = np.where((self.prev_mask > 1) & (z_hat == 0), 1, z_hat)

            self.prev_mask = z_hat.copy()
            warped_masks.append(z_hat.copy())
        return warped_masks

    def _create_selection_mask(
        self, err_forward_lst: list[np.ndarray], err_backward_lst: list[np.ndarray]
    ) -> list[np.ndarray]:
        err_forward = np.array(err_forward_lst)
        err_backward = np.array(err_backward_lst)

        if err_forward.shape != err_backward.shape:
            print(f"Shape mismatch: {err_forward.shape=} vs {err_backward.shape=}")
            return []

        # Create a binary mask where the forward error metric
        # is less than the backward error metric
        selection_masks = np.where(err_forward < err_backward, 0, 1).astype(np.uint8)

        # Convert numpy array back to list
        selection_masks_lst = [
            selection_masks[i] for i in range(selection_masks.shape[0])
        ]

        return selection_masks_lst

    def _hist_blend(
        self,
        style_fwd: list[np.ndarray],
        style_bwd: list[np.ndarray],
        err_masks: list[np.ndarray],
    ) -> list[np.ndarray]:
        st = time.time()
        hist_blends: list[np.ndarray] = []
        for i in tqdm.tqdm(range(len(err_masks)), desc="Hist blending: "):
            if self.use_gpu:
                hist_blend = hist_blend_cupy(
                    style_fwd[i],
                    style_bwd[i],
                    err_masks[i],
                )
            else:
                hist_blend = hist_blender(
                    style_fwd[i],
                    style_bwd[i],
                    err_masks[i],
                )
            hist_blends.append(hist_blend)
        print(f"Hist Blend took {time.time() - st:.4f} s")
        print(len(hist_blends))
        return hist_blends

    def _reconstruct(
        self,
        style_fwd: list[np.ndarray],
        style_bwd: list[np.ndarray],
        err_masks: list[np.ndarray],
        hist_blends: list[np.ndarray],
    ):
        blends = reconstructor(
            hist_blends,
            style_fwd,
            style_bwd,
            err_masks,
            use_gpu=self.use_gpu,
            use_lsqr=self.use_lsqr,
            use_poisson_cupy=self.use_poisson_cupy,
            poisson_maxiter=self.poisson_maxiter,
        )
        final_blends = blends._create()
        final_blends = [blend for blend in final_blends if blend is not None]
        return final_blends
