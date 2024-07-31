import numpy as np

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization.
    """
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3), dtype=np.uint8)
    col = 0

    colorwheel[col : col + RY, 0] = 255
    colorwheel[col : col + RY, 1] = np.floor(255 * np.arange(0, RY) / RY).astype(
        np.uint8
    )
    col += RY

    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG).astype(
        np.uint8
    )
    colorwheel[col : col + YG, 1] = 255
    col += YG

    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC).astype(
        np.uint8
    )
    col += GC

    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(0, CB) / CB).astype(
        np.uint8
    )
    colorwheel[col : col + CB, 2] = 255
    col += CB

    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM).astype(
        np.uint8
    )
    col += BM

    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(0, MR) / MR).astype(
        np.uint8
    )
    colorwheel[col : col + MR, 0] = 255

    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    """
    flow_image = np.zeros((*u.shape, 3), dtype=np.uint8)
    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % ncols
    f = fk - k0

    for i in range(3):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] *= 0.75

        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[..., ch_idx] = (255 * col).astype(np.uint8)

    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Converts a two-dimensional flow image to a color image for visualization.
    """
    assert (
        flow_uv.ndim == 3 and flow_uv.shape[2] == 2
    ), "Input flow must have shape [H,W,2]"

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u, v = flow_uv[..., 0], flow_uv[..., 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_uv_to_colors(u, v, convert_to_bgr)
