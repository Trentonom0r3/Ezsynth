import numpy as np
import torch
import torch.nn.functional as F
from scipy import interpolate


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, dims, mode="sintel"):
        self.ht, self.wd = dims[-2:]
        self._calculate_padding(mode)

    def _calculate_padding(self, mode):
        pad_ht = -self.ht % 8
        pad_wd = -self.wd % 8

        if mode == "sintel":
            self._pad = (
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            )
        else:
            self._pad = (pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht)

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        return x[
            ...,
            self._pad[2] : self.ht + self._pad[2],
            self._pad[0] : self.wd + self._pad[0],
        ]


def forward_interpolate(flow_ts):
    with torch.no_grad():
        flow = flow_ts.cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht), indexing="xy")

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 >= 0) & (x1 < wd) & (y1 >= 0) & (y1 < ht)

    x1_valid = x1[valid]
    y1_valid = y1[valid]
    dx_valid = dx[valid]
    dy_valid = dy[valid]

    flow_x = interpolate.griddata(
        (x1_valid, y1_valid), dx_valid, (x0, y0), method="nearest", fill_value=0
    )

    flow_y = interpolate.griddata(
        (x1_valid, y1_valid), dy_valid, (x0, y0), method="nearest", fill_value=0
    )

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(
        torch.arange(ht, device=device), torch.arange(wd, device=device), indexing="ij"
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode="bilinear"):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
