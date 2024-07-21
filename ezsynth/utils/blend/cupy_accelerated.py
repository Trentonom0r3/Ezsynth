import time

import cupy as cp
import cupyx
import cupyx.scipy.sparse._csc
import cupyx.scipy.sparse.linalg
import cv2
import numpy as np
import scipy.sparse


def assemble_min_error_img(a, b, error_mask):
    return cp.where(error_mask == 0, a, b)


def mean_std(img):
    return cp.mean(img, axis=(0, 1)), cp.std(img, axis=(0, 1))


def histogram_transform(img, means, stds, target_means, target_stds):
    return ((img - means) * target_stds / stds + target_means).astype(cp.float32)


def hist_blend_cupy(
    a: np.ndarray, b: np.ndarray, error_mask: np.ndarray, weight1=0.5, weight2=0.5
):
    a = cp.asarray(a)
    b = cp.asarray(b)
    error_mask = cp.asarray(error_mask)

    # Ensure error_mask has 3 channels
    if len(error_mask.shape) == 2:
        error_mask = cp.repeat(error_mask[:, :, cp.newaxis], 3, axis=2)

    # Convert to Lab color space
    a_lab = cv2.cvtColor(cp.asnumpy(a), cv2.COLOR_BGR2Lab)
    b_lab = cv2.cvtColor(cp.asnumpy(b), cv2.COLOR_BGR2Lab)
    a_lab = cp.asarray(a_lab)
    b_lab = cp.asarray(b_lab)

    min_error_lab = assemble_min_error_img(a_lab, b_lab, error_mask)

    # Compute means and stds
    a_mean, a_std = mean_std(a_lab)
    b_mean, b_std = mean_std(b_lab)
    min_error_mean, min_error_std = mean_std(min_error_lab)

    # Histogram transformation constants
    t_mean = cp.full(3, 0.5 * 256, dtype=cp.float32)
    t_std = cp.full(3, (1 / 36) * 256, dtype=cp.float32)

    # Histogram transform
    a_lab = histogram_transform(a_lab, a_mean, a_std, t_mean, t_std)
    b_lab = histogram_transform(b_lab, b_mean, b_std, t_mean, t_std)

    # Blending
    ab_lab = (a_lab * weight1 + b_lab * weight2 - 128) / 0.5 + 128
    ab_mean, ab_std = mean_std(ab_lab)

    # Final histogram transform
    ab_lab = histogram_transform(ab_lab, ab_mean, ab_std, min_error_mean, min_error_std)

    ab_lab = cp.clip(cp.round(ab_lab), 0, 255).astype(cp.uint8)
    ab_lab = cp.asnumpy(ab_lab)
    ab = cv2.cvtColor(ab_lab, cv2.COLOR_Lab2BGR)
    return ab


def construct_A_cupy(h: int, w: int, grad_weight: list[float], use_poisson_cupy=False):
    st = time.time()
    indgx_x = cp.zeros(2 * (h - 1) * w, dtype=int)
    indgx_y = cp.zeros(2 * (h - 1) * w, dtype=int)
    vdx = cp.ones(2 * (h - 1) * w)

    indgy_x = cp.zeros(2 * h * (w - 1), dtype=int)
    indgy_y = cp.zeros(2 * h * (w - 1), dtype=int)
    vdy = cp.ones(2 * h * (w - 1))

    indgx_x[::2] = cp.arange((h - 1) * w)
    indgx_y[::2] = indgx_x[::2]
    indgx_x[1::2] = indgx_x[::2]
    indgx_y[1::2] = indgx_x[::2] + w

    indgy_x[::2] = cp.arange(h * (w - 1))
    indgy_y[::2] = indgy_x[::2]
    indgy_x[1::2] = indgy_x[::2]
    indgy_y[1::2] = indgy_x[::2] + 1

    vdx[1::2] = -1
    vdy[1::2] = -1

    Ix = cupyx.scipy.sparse.eye(h * w, format="csc")
    Gx = cupyx.scipy.sparse.coo_matrix(
        (vdx, (indgx_x, indgx_y)), shape=(h * w, h * w)
    ).tocsc()
    Gy = cupyx.scipy.sparse.coo_matrix(
        (vdy, (indgy_x, indgy_y)), shape=(h * w, h * w)
    ).tocsc()

    As = [
        cupyx.scipy.sparse.vstack([Gx * weight, Gy * weight, Ix])
        for weight in grad_weight
    ]
    print(f"Constructing As took {time.time() - st:.4f} s")
    if not use_poisson_cupy:
        As_scipy = [
            scipy.sparse.vstack(
                [
                    scipy.sparse.csr_matrix(Gx.get() * weight),
                    scipy.sparse.csr_matrix(Gy.get() * weight),
                    scipy.sparse.csr_matrix(Ix.get()),
                ]
            )
            for weight in grad_weight
        ]
        return As_scipy
    return As


def poisson_fusion_cupy(
    blendI: np.ndarray,
    I1: np.ndarray,
    I2: np.ndarray,
    mask: np.ndarray,
    As: list[cupyx.scipy.sparse._csc.csc_matrix],
    poisson_maxiter=None,
):
    grad_weight = [2.5, 0.5, 0.5]
    Iab = cv2.cvtColor(blendI, cv2.COLOR_BGR2LAB).astype(float)
    Ia = cv2.cvtColor(I1, cv2.COLOR_BGR2LAB).astype(float)
    Ib = cv2.cvtColor(I2, cv2.COLOR_BGR2LAB).astype(float)

    Iab_cp = cp.asarray(Iab)
    Ia_cp = cp.asarray(Ia)
    Ib_cp = cp.asarray(Ib)
    mask_cp = cp.asarray(mask)

    m_cp = (mask_cp > 0).astype(float)[..., cp.newaxis]
    h, w, c = Iab.shape

    gx_cp = cp.zeros_like(Ia_cp)
    gy_cp = cp.zeros_like(Ia_cp)

    gx_cp[:-1] = (Ia_cp[:-1] - Ia_cp[1:]) * (1 - m_cp[:-1]) + (
        Ib_cp[:-1] - Ib_cp[1:]
    ) * m_cp[:-1]
    gy_cp[:, :-1] = (Ia_cp[:, :-1] - Ia_cp[:, 1:]) * (1 - m_cp[:, :-1]) + (
        Ib_cp[:, :-1] - Ib_cp[:, 1:]
    ) * m_cp[:, :-1]

    final_channels = [
        poisson_fusion_channel_cupy(
            Iab_cp, gx_cp, gy_cp, h, w, As, i, grad_weight, maxiter=poisson_maxiter
        )
        for i in range(3)
    ]

    final = np.clip(np.concatenate(final_channels, axis=2), 0, 255)
    return cv2.cvtColor(final.astype(np.uint8), cv2.COLOR_LAB2BGR)


def poisson_fusion_channel_cupy(
    Iab: cp.ndarray,
    gx: cp.ndarray,
    gy: cp.ndarray,
    h: int,
    w: int,
    As: list[cupyx.scipy.sparse._csc.csc_matrix],
    channel: int,
    grad_weight: list[float],
    maxiter: int | None = None,
):
    cp.get_default_memory_pool().free_all_blocks()
    weight = grad_weight[channel]
    im_dx = cp.clip(gx[:, :, channel].reshape(h * w, 1), -100, 100)
    im_dy = cp.clip(gy[:, :, channel].reshape(h * w, 1), -100, 100)
    im = Iab[:, :, channel].reshape(h * w, 1)
    im_mean = im.mean()
    im = im - im_mean
    A = As[channel]
    b = cp.vstack([im_dx * weight, im_dy * weight, im])
    out = cupyx.scipy.sparse.linalg.lsmr(A, b, maxiter=maxiter)
    out_im = (out[0] + im_mean).reshape(h, w, 1)

    return cp.asnumpy(out_im)
