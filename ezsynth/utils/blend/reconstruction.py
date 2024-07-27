import time

import cv2
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import tqdm

try:
    from .cupy_accelerated import construct_A_cupy, poisson_fusion_cupy

    USE_GPU = True
    print("Cupy is installed. Can do Cupy GPU accelerations")
except ImportError:
    USE_GPU = False
    print("Cupy is not installed. Revert to CPU")


class reconstructor:
    def __init__(
        self,
        hist_blends: list[np.ndarray],
        style_fwd: list[np.ndarray],
        style_bwd: list[np.ndarray],
        err_masks: list[np.ndarray],
        use_gpu=False,
        use_lsqr=True,
        use_poisson_cupy=False,
        poisson_maxiter=None,
    ):
        self.hist_blends = hist_blends
        self.style_fwd = style_fwd
        self.style_bwd = style_bwd
        self.err_masks = err_masks
        self.blends = None

        self.use_gpu = use_gpu and USE_GPU
        self.use_lsqr = use_lsqr
        self.use_poisson_cupy = self.use_gpu and use_poisson_cupy
        self.poisson_maxiter = poisson_maxiter

    def _create(self):
        num_blends = len(self.hist_blends)
        h, w, c = self.hist_blends[0].shape
        self.blends = np.zeros((num_blends, h, w, c))

        a = construct_A(h, w, [2.5, 0.5, 0.5], self.use_gpu, self.use_poisson_cupy)
        for i in tqdm.tqdm(range(num_blends)):
            self.blends[i] = poisson_fusion(
                self.hist_blends[i],
                self.style_fwd[i],
                self.style_bwd[i],
                self.err_masks[i],
                a,
                self.use_gpu,
                self.use_lsqr,
                self.use_poisson_cupy,
                self.poisson_maxiter,
            )

        return self.blends


def construct_A(
    h: int, w: int, grad_weight: list[float], use_gpu=False, use_poisson_cupy=False
):
    if use_gpu:
        return construct_A_cupy(h, w, grad_weight, use_poisson_cupy)
    return construct_A_cpu(h, w, grad_weight)


def poisson_fusion(
    blendI: np.ndarray,
    I1: np.ndarray,
    I2: np.ndarray,
    mask: np.ndarray,
    As,
    use_gpu=False,
    use_lsqr=True,
    use_poisson_cupy=False,
    poisson_maxiter=None,
):
    if use_gpu and use_poisson_cupy:
        return poisson_fusion_cupy(blendI, I1, I2, mask, As, poisson_maxiter)
    return poisson_fusion_cpu_optimized(
        blendI, I1, I2, mask, As, use_lsqr, poisson_maxiter
    )


def construct_A_cpu(h: int, w: int, grad_weight: list[float]):
    st = time.time()
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

    Ix = scipy.sparse.eye(h * w, format="csc")
    Gx = scipy.sparse.coo_matrix(
        (vdx, (indgx_x, indgx_y)), shape=(h * w, h * w)
    ).tocsc()
    Gy = scipy.sparse.coo_matrix(
        (vdy, (indgy_x, indgy_y)), shape=(h * w, h * w)
    ).tocsc()

    As = [scipy.sparse.vstack([Gx * weight, Gy * weight, Ix]) for weight in grad_weight]
    print(f"Constructing As took {time.time() - st:.4f} s")
    return As


def poisson_fusion_cpu(
    blendI: np.ndarray,
    I1: np.ndarray,
    I2: np.ndarray,
    mask: np.ndarray,
    As: list[scipy.sparse._csc.csc_matrix],
    use_lsqr=True,
):
    grad_weight = [2.5, 0.5, 0.5]
    Iab = cv2.cvtColor(blendI, cv2.COLOR_BGR2LAB).astype(float)
    Ia = cv2.cvtColor(I1, cv2.COLOR_BGR2LAB).astype(float)
    Ib = cv2.cvtColor(I2, cv2.COLOR_BGR2LAB).astype(float)

    m = (mask > 0).astype(float)[..., np.newaxis]
    h, w, c = Iab.shape

    gx = np.zeros_like(Ia)
    gy = np.zeros_like(Ia)

    gx[:-1] = (Ia[:-1] - Ia[1:]) * (1 - m[:-1]) + (Ib[:-1] - Ib[1:]) * m[:-1]
    gy[:, :-1] = (Ia[:, :-1] - Ia[:, 1:]) * (1 - m[:, :-1]) + (
        Ib[:, :-1] - Ib[:, 1:]
    ) * m[:, :-1]

    final_channels = [
        poisson_fusion_channel_cpu(Iab, gx, gy, h, w, As, i, grad_weight, use_lsqr)
        for i in range(3)
    ]

    final = np.clip(np.concatenate(final_channels, axis=2), 0, 255)
    return cv2.cvtColor(final.astype(np.uint8), cv2.COLOR_LAB2BGR)


def poisson_fusion_channel_cpu(
    Iab: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    h: int,
    w: int,
    As: list[scipy.sparse._csc.csc_matrix],
    channel: int,
    grad_weight: list[float],
    use_lsqr=True,
):
    """Helper function to perform Poisson fusion on a single channel."""
    weight = grad_weight[channel]
    im_dx = np.clip(gx[:, :, channel].reshape(h * w, 1), -100, 100)
    im_dy = np.clip(gy[:, :, channel].reshape(h * w, 1), -100, 100)
    im = Iab[:, :, channel].reshape(h * w, 1)
    im_mean = im.mean()
    im = im - im_mean

    A = As[channel]
    b = np.vstack([im_dx * weight, im_dy * weight, im])
    if use_lsqr:
        out = scipy.sparse.linalg.lsqr(A, b)
    else:
        out = scipy.sparse.linalg.lsmr(A, b)
    out_im = (out[0] + im_mean).reshape(h, w, 1)

    return out_im


def gradient_compute_python(Ia: np.ndarray, Ib: np.ndarray, m: np.ndarray):
    gx = np.zeros_like(Ia)
    gy = np.zeros_like(Ia)

    gx[:-1] = (Ia[:-1] - Ia[1:]) * (1 - m[:-1]) + (Ib[:-1] - Ib[1:]) * m[:-1]
    gy[:, :-1] = (Ia[:, :-1] - Ia[:, 1:]) * (1 - m[:, :-1]) + (
        Ib[:, :-1] - Ib[:, 1:]
    ) * m[:, :-1]
    return gx, gy


def poisson_fusion_cpu_optimized(
    blendI: np.ndarray,
    I1: np.ndarray,
    I2: np.ndarray,
    mask: np.ndarray,
    As: list[scipy.sparse._csc.csc_matrix],
    use_lsqr=True,
    poisson_maxiter=None,
):
    # grad_weight = [2.5, 0.5, 0.5]
    grad_weight = np.array([2.5, 0.5, 0.5])
    Iab = cv2.cvtColor(blendI, cv2.COLOR_BGR2LAB).astype(float)
    Ia = cv2.cvtColor(I1, cv2.COLOR_BGR2LAB).astype(float)
    Ib = cv2.cvtColor(I2, cv2.COLOR_BGR2LAB).astype(float)

    m = (mask > 0).astype(float)[..., np.newaxis]
    h, w, c = Iab.shape

    gx, gy = gradient_compute_python(Ia, Ib, m)

    # Reshape and clip all channels at once
    gx_reshaped = np.clip(gx.reshape(h * w, c), -100, 100)
    gy_reshaped = np.clip(gy.reshape(h * w, c), -100, 100)

    Iab_reshaped = Iab.reshape(h * w, c)
    Iab_mean = np.mean(Iab_reshaped, axis=0)
    Iab_centered = Iab_reshaped - Iab_mean

    # Pre-allocate the output array
    out_all = np.zeros((h * w, c), dtype=np.float32)

    for channel in range(3):
        weight = grad_weight[channel]
        im_dx = gx_reshaped[:, channel : channel + 1]
        im_dy = gy_reshaped[:, channel : channel + 1]
        im = Iab_centered[:, channel : channel + 1]

        A = As[channel]
        b = np.vstack([im_dx * weight, im_dy * weight, im])

        if use_lsqr:
            out_all[:, channel] = scipy.sparse.linalg.lsqr(A, b)[0]
        else:
            out_all[:, channel] = scipy.sparse.linalg.lsmr(
                A, b, maxiter=poisson_maxiter
            )[0]

    # Add back the mean and reshape
    final = (out_all + Iab_mean).reshape(h, w, c)
    final = np.clip(final, 0, 255)
    return cv2.cvtColor(final.astype(np.uint8), cv2.COLOR_LAB2BGR)
