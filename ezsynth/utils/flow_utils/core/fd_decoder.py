import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .corr import CorrBlock

# --- transformer modules
class TransformerModule(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tb = TransBlocks(args)

    def forward(self, fmap1, fmap2, inp):
        batch, ch, ht, wd = fmap1.shape
        fmap1, fmap2 = self.tb(fmap1, fmap2)
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.args.corr_levels, radius=self.args.corr_radius)
        return corr_fn


class TransBlocks(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args.m_dim
        mlp_scale = 4
        window_size = [8, 8]
        num_layers = [2, 2]

        self.num_layers = len(num_layers)
        self.blocks = nn.ModuleList()
        for n in range(self.num_layers):
            if n == self.num_layers - 1:
                self.blocks.append(
                    BasicLayer(num_layer=num_layers[n], dim=dim, mlp_scale=mlp_scale, window_size=window_size, cross=False))
            else:
                self.blocks.append(
                    BasicLayer(num_layer=num_layers[n], dim=dim, mlp_scale=mlp_scale, window_size=window_size, cross=True))

    def forward(self, fmap1, fmap2):
        _, _, ht, wd = fmap1.shape
        pad_h, pad_w = (8 - (ht % 8)) % 8, (8 - (wd % 8)) % 8
        _pad = [pad_w // 2, pad_w - pad_w // 2, pad_h, 0]
        fmap1 = F.pad(fmap1, pad=_pad, mode='constant', value=0)
        fmap2 = F.pad(fmap2, pad=_pad, mode='constant', value=0)
        mask = torch.zeros([1, ht, wd]).to(fmap1.device)
        mask = torch.nn.functional.pad(mask, pad=_pad, mode='constant', value=1)
        mask = mask.bool()
        fmap1 = fmap1.permute(0, 2, 3, 1).contiguous().float()
        fmap2 = fmap2.permute(0, 2, 3, 1).contiguous().float()

        for idx, blk in enumerate(self.blocks):
            fmap1, fmap2 = blk(fmap1, fmap2, mask=mask)

        _, ht, wd, _ = fmap1.shape
        fmap1 = fmap1[:, _pad[2]:ht - _pad[3], _pad[0]:wd - _pad[1], :]
        fmap2 = fmap2[:, _pad[2]:ht - _pad[3], _pad[0]:wd - _pad[1], :]

        fmap1 = fmap1.permute(0, 3, 1, 2).contiguous()
        fmap2 = fmap2.permute(0, 3, 1, 2).contiguous()

        return fmap1, fmap2


def window_partition(fmap, window_size):
    """
    :param fmap: shape:B, H, W, C
    :param window_size: Wh, Ww
    :return: shape: B*nW, Wh*Ww, C
    """
    B, H, W, C = fmap.shape
    fmap = fmap.reshape(B, H//window_size[0], window_size[0], W//window_size[1], window_size[1], C)
    fmap = fmap.permute(0, 1, 3, 2, 4, 5).contiguous()
    fmap = fmap.reshape(B*(H//window_size[0])*(W//window_size[1]), window_size[0]*window_size[1], C)
    return fmap


def window_reverse(fmap, window_size, H, W):
    """
    :param fmap: shape:B*nW, Wh*Ww, dim
    :param window_size: Wh, Ww
    :param H: original image height
    :param W: original image width
    :return: shape: B, H, W, C
    """
    Bnw, _, dim = fmap.shape
    nW = (H // window_size[0]) * (W // window_size[1])
    fmap = fmap.reshape(Bnw//nW, H // window_size[0], W // window_size[1], window_size[0], window_size[1], dim)
    fmap = fmap.permute(0, 1, 3, 2, 4, 5).contiguous()
    fmap = fmap.reshape(Bnw//nW, H, W, dim)
    return fmap


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, scale=None):
        super().__init__()
        self.dim = dim
        self.scale = scale or dim ** (-0.5)
        self.q = nn.Linear(in_features=dim, out_features=dim)
        self.k = nn.Linear(in_features=dim, out_features=dim)
        self.v = nn.Linear(in_features=dim, out_features=dim)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, fmap, mask=None):
        """
        :param fmap1: B*nW, Wh*Ww, dim
        :param mask: nw, Wh*Ww, Ww*Wh
        :return: B*nW, Wh*Ww, dim
        """
        Bnw, WhWw, dim = fmap.shape
        q = self.q(fmap)
        k = self.k(fmap)
        v = self.v(fmap)

        q = q * self.scale
        attn = q @ k.transpose(1, 2)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.reshape(Bnw//nw, nw, WhWw, WhWw) + mask.unsqueeze(0)
            attn = attn.reshape(Bnw, WhWw, WhWw)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        x = attn @ v
        x = self.proj(x)
        return x


class GlobalAttention(nn.Module):
    def __init__(self, dim, scale=None):
        super().__init__()
        self.dim = dim
        self.scale = scale or dim ** (-0.5)
        self.q = nn.Linear(in_features=dim, out_features=dim)
        self.k = nn.Linear(in_features=dim, out_features=dim)
        self.v = nn.Linear(in_features=dim, out_features=dim)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(in_features=dim, out_features=dim)

    def forward(self, fmap1, fmap2, mask=None):
        """
        :param fmap1: B, H, W, C
        :param fmap2: B, H, W, C
        :param pe: B, H, W, C
        :return:
        """
        B, H, W, C = fmap1.shape
        q = self.q(fmap1)
        k = self.k(fmap2)
        v = self.v(fmap2)

        q, k, v = map(lambda x: x.reshape(B, H*W, C), [q, k, v])

        q = q * self.scale
        attn = q @ k.transpose(1, 2)
        if mask is not None:
            mask = mask.reshape(1, H * W, 1) | mask.reshape(1, 1, H * W)  # batch, hw, hw
            mask = mask.float() * -100.0
            attn = attn + mask
        attn = self.softmax(attn)
        x = attn @ v  # B, HW, C

        x = self.proj(x)

        x = x.reshape(B, H, W, C)
        return x


class SelfTransformerBlcok(nn.Module):
    def __init__(self, dim, mlp_scale, window_size, shift_size=None, norm=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        if norm == 'layer':
            self.layer_norm1 = nn.LayerNorm(dim)
            self.layer_norm2 = nn.LayerNorm(dim)
        else:
            self.layer_norm1 = nn.Identity()
            self.layer_norm2 = nn.Identity()

        self.self_attn = WindowAttention(dim=dim, window_size=window_size)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_scale),
            nn.GELU(),
            nn.Linear(dim * mlp_scale, dim)
        )

    def forward(self, fmap, mask=None):
        """
        :param fmap: shape: B, H, W, C
        :return: B, H, W, C
        """
        B, H, W, C = fmap.shape

        shortcut = fmap
        fmap = self.layer_norm1(fmap)

        if self.shift_size is not None:
            shifted_fmap = torch.roll(fmap, [-self.shift_size[0], -self.shift_size[1]], dims=(1, 2))
            if mask is not None:
                shifted_mask = torch.roll(mask, [-self.shift_size[0], -self.shift_size[1]], dims=(1, 2))
        else:
            shifted_fmap = fmap
            if mask is not None:
                shifted_mask = mask

        win_fmap = window_partition(shifted_fmap, window_size=self.window_size)
        if mask is not None:
            pad_mask = window_partition(shifted_mask.unsqueeze(-1), self.window_size)
            pad_mask = pad_mask.reshape(-1, self.window_size[0] * self.window_size[1], 1) \
                       | pad_mask.reshape(-1, 1, self.window_size[0] * self.window_size[1])

        if self.shift_size is not None:
            h_slice = [slice(0, -self.window_size[0]), slice(-self.window_size[0], -self.shift_size[0]), slice(-self.shift_size[0], None)]
            w_slice = [slice(0, -self.window_size[1]), slice(-self.window_size[1], -self.shift_size[1]), slice(-self.shift_size[1], None)]
            img_mask = torch.zeros([1, H, W, 1]).to(win_fmap.device)
            count = 0
            for h in h_slice:
                for w in w_slice:
                    img_mask[:, h, w, :] = count
                    count += 1
            win_mask = window_partition(img_mask, self.window_size)
            win_mask = win_mask.reshape(-1, self.window_size[0] * self.window_size[1])  # nW, Wh*Ww
            attn_mask = win_mask.unsqueeze(2) - win_mask.unsqueeze(1)  # nw, Wh*Ww, Wh*Ww
            if mask is not None:
                attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0).masked_fill((attn_mask != 0) | pad_mask, -100.0)
            else:
                attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0).masked_fill(attn_mask != 0, -100.0)
            attn_fmap = self.self_attn(win_fmap, attn_mask)
        else:
            if mask is not None:
                pad_mask = pad_mask.float()
                pad_mask = pad_mask.masked_fill(pad_mask != 0, -100.0).masked_fill(pad_mask == 0, 0.0)
                attn_fmap = self.self_attn(win_fmap, pad_mask)
            else:
                attn_fmap = self.self_attn(win_fmap, None)
        shifted_fmap = window_reverse(attn_fmap, self.window_size, H, W)

        if self.shift_size is not None:
            fmap = torch.roll(shifted_fmap, [self.shift_size[0], self.shift_size[1]], dims=(1, 2))
        else:
            fmap = shifted_fmap

        fmap = shortcut + fmap
        fmap = fmap + self.mlp(self.layer_norm2(fmap))  # B, H, W, C
        return fmap


class CrossTransformerBlcok(nn.Module):
    def __init__(self, dim, mlp_scale, norm=None):
        super().__init__()
        self.dim = dim

        if norm == 'layer':
            self.layer_norm1 = nn.LayerNorm(dim)
            self.layer_norm2 = nn.LayerNorm(dim)
            self.layer_norm3 = nn.LayerNorm(dim)
        else:
            self.layer_norm1 = nn.Identity()
            self.layer_norm2 = nn.Identity()
            self.layer_norm3 = nn.Identity()
        self.cross_attn = GlobalAttention(dim=dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_scale),
            nn.GELU(),
            nn.Linear(dim * mlp_scale, dim)
        )

    def forward(self, fmap1, fmap2, mask=None):
        """
        :param fmap1: shape: B, H, W, C
        :param fmap2: shape: B, H, W, C
        :return: B, H, W, C
        """
        shortcut = fmap1

        fmap1 = self.layer_norm1(fmap1)
        fmap2 = self.layer_norm2(fmap2)

        attn_fmap = self.cross_attn(fmap1, fmap2, mask)
        attn_fmap = shortcut + attn_fmap
        fmap = attn_fmap + self.mlp(self.layer_norm3(attn_fmap))  # B, H, W, C
        return fmap


class BasicLayer(nn.Module):
    def __init__(self, num_layer, dim, mlp_scale, window_size, cross=False):
        super().__init__()
        assert num_layer % 2 == 0, "The number of Transformer Block must be even!"
        self.blocks = nn.ModuleList()
        for n in range(num_layer):
            shift_size = None if n % 2 == 0 else [window_size[0]//2, window_size[1]//2]
            self.blocks.append(
                SelfTransformerBlcok(
                    dim=dim,
                    mlp_scale=mlp_scale,
                    window_size=window_size,
                    shift_size=shift_size,
                    norm='layer'))
                    
        if cross:
            self.cross_transformer = CrossTransformerBlcok(dim=dim, mlp_scale=mlp_scale, norm='layer')

        self.cross = cross

    def forward(self, fmap1, fmap2, mask=None):
        """
        :param fmap1: B, H, W, C
        :param fmap2: B, H, W, C
        :return: B, H, W, C
        """
        B = fmap1.shape[0]
        fmap = torch.cat([fmap1, fmap2], dim=0)
        for blk in self.blocks:
            fmap = blk(fmap, mask)
        fmap1, fmap2 = torch.split(fmap, [B]*2, dim=0)
        if self.cross:
            fmap2 = self.cross_transformer(fmap2, fmap1, mask) + fmap2
            fmap1 = self.cross_transformer(fmap1, fmap2, mask) + fmap1
        return fmap1, fmap2


# --- upsample modules
class UpSampleMask8(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up_sample_mask = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=64 * 9, kernel_size=1, stride=1)
        )

    def forward(self, data):
        """
        :param data:  B, C, H, W
        :return:  batch, 8*8*9, H, W
        """
        mask = self.up_sample_mask(data)  # B, 64*6, H, W
        return mask


class UpSampleMask4(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up_sample_mask = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=16 * 9, kernel_size=1, stride=1)
        )

    def forward(self, data):
        """
        :param data:  B, C, H, W
        :return:  batch, 8*8*9, H, W
        """
        mask = self.up_sample_mask(data)  # B, 64*6, H, W
        return mask


# --- SK decoder modules
class PCBlock4_Deep_nopool_res(nn.Module):
    def __init__(self, C_in, C_out, k_conv):
        super().__init__()
        self.conv_list = nn.ModuleList([
            nn.Conv2d(C_in, C_in, kernel, stride=1, padding=kernel//2, groups=C_in) for kernel in k_conv])

        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5*C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5*C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = self.ffn2(x)
        return x


class SKMotionEncoder6_Deep_nopool_res(nn.Module):
    def __init__(self, args):
        super().__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = PCBlock4_Deep_nopool_res(cor_planes, 256, k_conv=args.k_conv)
        self.convc2 = PCBlock4_Deep_nopool_res(256, 192, k_conv=args.k_conv)

        self.convf1 = nn.Conv2d(2, 128, 1, 1, 0)
        self.convf2 = PCBlock4_Deep_nopool_res(128, 64, k_conv=args.k_conv)

        self.conv = PCBlock4_Deep_nopool_res(64+192, 128-2, k_conv=args.k_conv)

    def forward(self, flow, corr):
        cor = F.gelu(self.convc1(corr))

        cor = self.convc2(cor)

        flo = self.convf1(flow)
        flo = self.convf2(flo)

        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.conv(cor_flo)

        return torch.cat([out, flow], dim=1)


# class SKUpdateBlock6_Deep_nopoolres_AllDecoder(nn.Module):
#     def __init__(self, args, hidden_dim):
#         super().__init__()
#         self.args = args
#         self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
#         self.gru = PCBlock4_Deep_nopool_res(128+hidden_dim+hidden_dim+128, 128, k_conv=args.PCUpdater_conv)
#         self.flow_head = PCBlock4_Deep_nopool_res(128, 2, k_conv=args.k_conv)

#         self.mask = nn.Sequential(
#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 64*9, 1, padding=0))

#         self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=self.args.num_heads)

#     def forward(self, net, inp, corr, flow, attention):
#         motion_features = self.encoder(flow, corr)
#         motion_features_global = self.aggregator(attention, motion_features)
#         inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

#         # Attentional update
#         net = self.gru(torch.cat([net, inp_cat], dim=1))

#         delta_flow = self.flow_head(net)

#         # scale mask to balence gradients
#         mask = .25 * self.mask(net)
#         return net, mask, delta_flow


class Aggregator(nn.Module):
    def __init__(self, args, chnn, heads=1):
        super().__init__()
        self.scale = chnn ** -0.5
        self.to_qk = nn.Conv2d(chnn, chnn * 2, 1, bias=False)
        self.to_v = nn.Conv2d(128, 128, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, *inputs):
        feat_ctx, feat_mo, itr = inputs

        feat_shape = feat_mo.shape
        b, c, h, w = feat_shape
        c_c = feat_ctx.shape[1]
        
        if itr == 0:
            feat_q, feat_k = self.to_qk(feat_ctx).chunk(2, dim=1)
            feat_q = self.scale * feat_q.view(b, c_c, h*w)
            feat_k = feat_k.view(b, c_c, h*w)

            attn = torch.einsum('b c n, b c m -> b m n', feat_q, feat_k)
            attn = attn.view(b, 1, h*w, h*w)
            self.attn = attn.softmax(2).view(b, h*w, h*w).permute(0, 2, 1).contiguous()

        feat_v = self.to_v(feat_mo).view(b, c, h*w)
        feat_o = torch.einsum('b n m, b c m -> b c n', self.attn, feat_v).contiguous().view(b, c, h, w)
        feat_o = feat_mo + feat_o * self.gamma
        return feat_o


class SKUpdate(nn.Module):
    def __init__(self, args, hidden_dim):
        super().__init__()
        self.args = args
        d_dim = args.c_dim 

        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        self.gru = PCBlock4_Deep_nopool_res(128+hidden_dim+hidden_dim+128, d_dim, k_conv=args.PCUpdater_conv)  
        self.flow_head = PCBlock4_Deep_nopool_res(d_dim, 2, k_conv=args.k_conv)
        self.aggregator = Aggregator(self.args, d_dim)

    def forward(self, net, inp, corr, flow, itr=None, sp4=False):
        motion_features = self.encoder(flow, corr)

        if not sp4:
            motion_features_global = self.aggregator(inp, motion_features, itr) 
        else:
            motion_features_global = motion_features

        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)
        net = self.gru(torch.cat([net, inp_cat], dim=1))

        delta_flow = self.flow_head(net)
        return net, delta_flow


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvEE(nn.Module):
    def __init__(self, C_in, C_out):
        super().__init__()
        groups = 4
        self.conv1 = nn.Sequential(
            nn.GroupNorm(groups, C_in),  
            nn.GELU(),
            nn.Conv2d(C_in, C_in, 3, padding=1),
            nn.GroupNorm(groups, C_in))
        self.conv2 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(C_in, C_in, 3, padding=1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, t_emb):
        scale, shift = t_emb
        x_res = x
        x = self.conv1(x)

        x = x * (scale + 1) + shift

        x = self.conv2(x)
        x_o = x * self.gamma

        return x_o


class SKUpdateDFM(nn.Module):
    def __init__(self, args, hidden_dim):
        super().__init__()
        self.args = args
        chnn = hidden_dim
        self.conv_ee = ConvEE(chnn, chnn)

        d_model = 256
        self.d_model = d_model
        time_dim = d_model * 2
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim))
        self.chnn_o = 256
        self.block_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, self.chnn_o))

    def forward(self, net, inp, corr, flow, itr, first_step=False, dfm_params=[]):
        t, funcs, i_ddim, dfm_itrs = dfm_params
        b = t.shape[0]
        time_emb = self.time_mlp(t)

        scale_shift = self.block_time_mlp(time_emb)
        scale_shift = scale_shift.view(b, 256, 1, 1)
        scale, shift = scale_shift.chunk(2, dim=1)

        motion_features = funcs.encoder(flow, corr)

        if first_step:
            self.shape = net.shape

        if self.shape == net.shape:
            feat_mo = funcs.aggregator(inp, motion_features, itr)
        else:
            feat_mo = motion_features

        feat_mo = self.conv_ee(feat_mo, [scale, shift])

        inp = torch.cat([inp, motion_features, feat_mo], dim=1)
        net = funcs.gru(torch.cat([net, inp], dim=1))

        net = net * (scale + 1) + shift

        delta_flow = funcs.flow_head(net)

        return net, delta_flow



