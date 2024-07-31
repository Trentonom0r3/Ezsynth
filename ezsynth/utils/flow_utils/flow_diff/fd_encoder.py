import torch
import torch.nn as nn
# import torch.nn.functional as F

import timm
import numpy as np

    
class twins_svt_large(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.svt = timm.create_model('twins_svt_large', pretrained=pretrained)

        del self.svt.head
        del self.svt.patch_embeds[2]
        del self.svt.patch_embeds[2]
        del self.svt.blocks[2]
        del self.svt.blocks[2]
        del self.svt.pos_block[2]
        del self.svt.pos_block[2]
    
    def forward(self, x, data=None, layer=2):
        B = x.shape[0]
        x_4 = None
        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):

            patch_size = embed.patch_size
            if i == layer - 1:
                embed.patch_size = (1, 1)
                embed.proj.stride = embed.patch_size
                x_4 = torch.nn.functional.pad(x, [1, 0, 1, 0], mode='constant', value=0)
                x_4, size_4 = embed(x_4)
                size_4 = (size_4[0] - 1, size_4[1] - 1)
                x_4 = drop(x_4)
                for j, blk in enumerate(blocks):
                    x_4 = blk(x_4, size_4)
                    if j == 0:
                        x_4 = pos_blk(x_4, size_4)

                if i < len(self.svt.depths) - 1:
                    x_4 = x_4.reshape(B, *size_4, -1).permute(0, 3, 1, 2).contiguous()

            embed.patch_size = patch_size
            embed.proj.stride = patch_size
            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j==0:
                    x = pos_blk(x, size)
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
            
            if i == layer-1:
                break
        
        return x, x_4

    def compute_params(self, layer=2):
        num = 0
        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):

            for param in embed.parameters():
                num +=  np.prod(param.size())

            for param in drop.parameters():
                num +=  np.prod(param.size())

            for param in blocks.parameters():
                num +=  np.prod(param.size())

            for param in pos_blk.parameters():
                num +=  np.prod(param.size())

            if i == layer-1:
                break

        for param in self.svt.head.parameters():
            num +=  np.prod(param.size())
        
        return num


class twins_svt_small_context(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.svt = timm.create_model('twins_svt_small', pretrained=pretrained)

        del self.svt.head
        del self.svt.patch_embeds[2]
        del self.svt.patch_embeds[2]
        del self.svt.blocks[2]
        del self.svt.blocks[2]
        del self.svt.pos_block[2]
        del self.svt.pos_block[2]

    def forward(self, x, data=None, layer=2):
        B = x.shape[0]
        x_4 = None
        for i, (embed, drop, blocks, pos_blk) in enumerate(
                zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):

            patch_size = embed.patch_size
            if i == layer - 1:
                embed.patch_size = (1, 1)
                embed.proj.stride = embed.patch_size
                x_4 = torch.nn.functional.pad(x, [1, 0, 1, 0], mode='constant', value=0)
                x_4, size_4 = embed(x_4)
                size_4 = (size_4[0] - 1, size_4[1] - 1)
                x_4 = drop(x_4)
                for j, blk in enumerate(blocks):
                    x_4 = blk(x_4, size_4)
                    if j == 0:
                        x_4 = pos_blk(x_4, size_4)

                if i < len(self.svt.depths) - 1:
                    x_4 = x_4.reshape(B, *size_4, -1).permute(0, 3, 1, 2).contiguous()

            embed.patch_size = patch_size
            embed.proj.stride = patch_size
            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()

            if i == layer - 1:
                break

        return x, x_4

    def compute_params(self, layer=2):
        num = 0
        for i, (embed, drop, blocks, pos_blk) in enumerate(
                zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):

            for param in embed.parameters():
                num += np.prod(param.size())

            for param in drop.parameters():
                num += np.prod(param.size())

            for param in blocks.parameters():
                num += np.prod(param.size())

            for param in pos_blk.parameters():
                num += np.prod(param.size())

            if i == layer - 1:
                break

        for param in self.svt.head.parameters():
            num += np.prod(param.size())

        return num

