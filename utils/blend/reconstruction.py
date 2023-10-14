import os
import cv2
import numpy as np
import scipy.sparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

class reconstructor:
    """Wraps the Poisson Reconstruction functionality into a simple class.
    Class should use '__call__' methods for actual execution.
    Example:
        blends = Reconstructor(hist_blends, style_fwd, style_bwd, err_masks)
    """
    def __init__(self, hist_blends, style_fwd, style_bwd, err_masks):
        self.hist_blends = hist_blends
        self.style_fwd = style_fwd
        self.style_bwd = style_bwd
        self.err_masks = err_masks
        self.blends = []

    def __call__(self):
        return self._create()
    
    def _create(self):
        num_blends = len(self.hist_blends)
        h, w, c = self.hist_blends[0].shape
        
        # Initialize a zero array to store the blends
        self.blends = np.zeros((num_blends, h, w, c))
        
        # get the # of cores
        num_cores = os.cpu_count()
        workers = num_cores/3
        # round workers to nearest integer
        workers = int(round(workers))
        # Multi-processing
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(poisson_fusion, self.hist_blends, self.style_fwd, self.style_bwd, self.err_masks)
        
        for i, blend in enumerate(results):
            self.blends[i] = blend
            
        return self.blends

def construct_A(h, w, grad_weight):
    with threading.Lock():
        indgx_x = np.zeros(2 * (h - 1) * w, dtype=int)
        indgx_y = np.zeros(2 * (h - 1) * w, dtype=int)
        vdx = np.ones(2 * (h - 1) * w)

        indgy_x = np.zeros(2 * h * (w - 1), dtype=int)
        indgy_y = np.zeros(2 * h * (w - 1), dtype=int)
        vdy = np.ones(2 * h * (w - 1))

        indgx_x[::2] = np.arange((h - 1) * w)
        indgx_y[::2] = indgx_x[::2]
        indgx_x[1::2] = indgx_x[::2]
        indgx_y[1::2] = indgx_x[::2] + w

        indgy_x[::2] = np.arange(h * (w - 1))
        indgy_y[::2] = indgy_x[::2]
        indgy_x[1::2] = indgy_x[::2]
        indgy_y[1::2] = indgy_x[::2] + 1

        vdx[1::2] = -1
        vdy[1::2] = -1

        Ix = scipy.sparse.eye(h * w, format='csc')
        Gx = scipy.sparse.coo_matrix((vdx, (indgx_x, indgx_y)), shape=(h * w, h * w)).tocsc()
        Gy = scipy.sparse.coo_matrix((vdy, (indgy_x, indgy_y)), shape=(h * w, h * w)).tocsc()

        As = [scipy.sparse.vstack([Gx * weight, Gy * weight, Ix]) for weight in grad_weight]
        return As

def poisson_fusion(blendI, I1, I2, mask, grad_weight=[2.5, 0.5, 0.5]):
    with threading.Lock():
        Iab = cv2.cvtColor(blendI, cv2.COLOR_BGR2LAB).astype(float)
        Ia = cv2.cvtColor(I1, cv2.COLOR_BGR2LAB).astype(float)
        Ib = cv2.cvtColor(I2, cv2.COLOR_BGR2LAB).astype(float)
        m = (mask > 0).astype(float)[:, :, np.newaxis]
        h, w, c = Iab.shape
        
        As = construct_A(h, w, grad_weight)
        
        gx = np.zeros_like(Ia)
        gy = np.zeros_like(Ia)

        gx[:-1, :, :] = (Ia[:-1, :, :] - Ia[1:, :, :]) * (1 - m[:-1, :, :]) + (Ib[:-1, :, :] - Ib[1:, :, :]) * m[:-1, :, :]
        gy[:, :-1, :] = (Ia[:, :-1, :] - Ia[:, 1:, :]) * (1 - m[:, :-1, :]) + (Ib[:, :-1, :] - Ib[:, 1:, :]) * m[:, :-1, :]
        
        final_channels = []
        with threading.Lock():
            with ThreadPoolExecutor(max_workers=3) as executor:
                for i in range(3):
                    future = executor.submit(poisson_fusion_channel, Iab, gx, gy, h, w, As, i, grad_weight)
                    final_channels.append(future.result())
        
        final = np.clip(np.concatenate(final_channels, axis=2), 0, 255)
        return cv2.cvtColor(final.astype(np.uint8), cv2.COLOR_LAB2BGR)


def poisson_fusion_channel(Iab, gx, gy, h, w, As, channel, grad_weight):
    """Helper function to perform Poisson fusion on a single channel."""
    with threading.Lock():
        weight = grad_weight[channel]
        im_dx = np.clip(gx[:, :, channel].reshape(h * w, 1), -100, 100)
        im_dy = np.clip(gy[:, :, channel].reshape(h * w, 1), -100, 100)
        im = Iab[:, :, channel].reshape(h * w, 1)
        im_mean = im.mean()
        im = im - im_mean
        
        A = As[channel]
        b = np.vstack([im_dx * weight, im_dy * weight, im])
        
        out = scipy.sparse.linalg.lsqr(A, b)
        out_im = (out[0] + im_mean).reshape(h, w, 1)
        
        return out_im
    
    
    