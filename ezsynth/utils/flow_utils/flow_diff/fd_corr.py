import torch
import torch.nn.functional as F

from ezsynth.utils.flow_utils.core.utils.utils import bilinear_sampler


class CorrBlock_FD_Sp4:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, coords_init=None, rad=1):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        corr = CorrBlock_FD_Sp4.corr(fmap1, fmap2, coords_init, r=rad)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2, coords_init, r):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        # return corr  / torch.sqrt(torch.tensor(dim).float())

        coords = coords_init.permute(0, 2, 3, 1).contiguous()
        batch, h1, w1, _ = coords.shape

        corr = corr.view(batch*h1*w1, 1, h1, w1)

        dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
        dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
        delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1)

        centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2)
        delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
        coords_lvl = centroid_lvl + delta_lvl

        corr = bilinear_sampler(corr, coords_lvl)

        corr = corr.view(batch, h1, w1, 1, 2*r+1, 2*r+1)
        return corr.permute(0, 1, 2, 3, 5, 4).contiguous() / torch.sqrt(torch.tensor(dim).float())
