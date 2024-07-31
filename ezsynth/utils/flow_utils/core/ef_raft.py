import torch
import torch.nn as nn
import torch.nn.functional as F

from .corr import EF_AlternateCorrBlock, EF_CorrBlock
from .extractor import BasicEncoder, SmallEncoder, CoordinateAttention
from .update import BasicUpdateBlock, SmallUpdateBlock, LookupScaler
from .utils.utils import coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except Exception as e:
    # dummy autocast for PyTorch < 1.6
    print(e)

    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class EF_RAFT(nn.Module):
    def __init__(self, args):
        super(EF_RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if "dropout" not in self.args:
            self.args.dropout = 0

        if "alternate_corr" not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(
                output_dim=128, norm_fn="instance", dropout=args.dropout
            )
            self.coor_att = None
            self.cnet = SmallEncoder(
                output_dim=hdim + cdim, norm_fn="none", dropout=args.dropout
            )
            self.lookup_scaler = None
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(
                output_dim=256, norm_fn="instance", dropout=args.dropout
            )
            self.coor_att = CoordinateAttention(feature_size=256, enc_size=128) # New
            self.cnet = BasicEncoder(
                output_dim=hdim + cdim, norm_fn="batch", dropout=args.dropout
            )
            self.lookup_scaler = LookupScaler(
                input_dim=hdim, output_size=args.corr_levels # New
            )
            self.update_block = BasicUpdateBlock(
                self.args, hidden_dim=hdim, input_dim=cdim + args.corr_levels * 4
            )  # Updated

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(
        self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False
    ):
        """Estimate optical flow between pair of frames"""

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
            fmap1 = self.coor_att(fmap1)
            fmap2 = self.coor_att(fmap2)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = EF_AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = EF_CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, base_inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            base_inp = torch.relu(base_inp)

        coords0, coords1 = self.initialize_flow(image1)

        BATCH_N, fC, fH, fW = fmap1.shape
        corr_map = corr_fn.corr_map
        soft_corr_map = F.softmax(corr_map, dim=2) * F.softmax(corr_map, dim=1)

        if flow_init is not None:
            coords1 = coords1 + flow_init
        else:  # Use the global matching idea as an initialization.
            match_f, match_f_ind = soft_corr_map.max(dim=2)  # Forward matching.
            match_b, match_b_ind = soft_corr_map.max(dim=1)  # Backward matching.

            # Permute the backward softmax for match the forward.
            for i in range(BATCH_N):
                match_b_tmp = match_b[i, ...]
                match_b[i, ...] = match_b_tmp[match_f_ind[i, ...]]

            # Replace the identity mapping with the found matches.
            matched = (match_f - match_b) == 0
            coords_index = (
                torch.arange(fH * fW)
                .unsqueeze(0)
                .repeat(BATCH_N, 1)
                .to(soft_corr_map.device)
            )
            coords_index[matched] = match_f_ind[matched]

            # Convert the 1D mapping to a 2D one.
            coords_index = coords_index.reshape(BATCH_N, fH, fW)
            coords_x = coords_index % fW
            coords_y = coords_index // fW
            coords1 = torch.stack([coords_x, coords_y], dim=1).float()

        # Iterative update
        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()

            lookup_scalers = None
            if self.lookup_scaler is not None:
                lookup_scalers = self.lookup_scaler(base_inp, net)
                cat_lookup_scalers = lookup_scalers.view(
                    -1, lookup_scalers.shape[-1] * lookup_scalers.shape[-2], 1, 1
                )
                cat_lookup_scalers = cat_lookup_scalers.expand(
                    -1, -1, base_inp.shape[2], base_inp.shape[3]
                )
                inp = torch.cat([base_inp, cat_lookup_scalers], dim=1)

            corr = corr_fn(coords1, scalers=lookup_scalers)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
