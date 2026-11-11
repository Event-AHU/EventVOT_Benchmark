import numpy as np
import torch.nn as nn

from timm.models.layers import to_2tuple
import torch
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768, norm_layer=False, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # allow different input size
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class PatchEmbed_event(nn.Module):
    def __init__(self, in_chans=320, embed_dim=768, kernel_size=5, stride=1, flatten=True, norm_layer=False):
        super().__init__()
        self.pos_embedding = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=1, stride=1)
        self.in_chans = in_chans
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride)
        # self.proj2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(embed_dim) if norm_layer else nn.Identity()
        # self.attn = SelfAttn(768, 4, 3072, 0.1, 'relu')

    def forward(self, x):
        # allow different input size
        x = x.type(torch.cuda.FloatTensor)
        xyz = self.pos_embedding(x.squeeze(dim=1)[:, :3, :])
        xyz = F.relu(xyz)
        x = torch.cat([xyz, x.squeeze(dim=1)[:, 3:, :]], dim=1)
        B, N, C = x.shape        # 1 1 19 10000
        H = W = int(np.sqrt(N*C//self.in_chans))
        x = x.reshape(B, self.in_chans, H, W)       #  B 19 100 100
        x = self.proj(x)        # B 768 16 16
        # x = self.proj2(x)        # B 768 16 16

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC  768*256
        # x = self.attn(src1=x, pos_src1=pos_embed)
        x = self.norm(x)
        return x

def xcorr_depthwise(x, kernel):  # x, event_x
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(2)
    H = W = int(np.sqrt(x.size(1)))
    x = x.reshape(1, batch*channel, H, W)
    kernel = kernel.reshape(batch*channel, 1, H, W)
    corr_weight = F.conv2d(x, kernel, groups=batch*channel)
    # out = out.reshape(batch, H*W, channel)
    out = x * corr_weight
    return out.reshape(batch, H*W, channel)
