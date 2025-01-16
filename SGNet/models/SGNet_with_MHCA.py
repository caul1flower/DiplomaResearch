from models.common import *
import torch
import torch.nn as nn

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, **block_kwargs):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()

    def forward(self, x, cond, mask=None):
        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)
        cond = cond.flatten(2).transpose(1, 2)

        N = H * W
        first_dim = 1 if _xformers_available else B

        q = self.q_linear(x)
        kv = self.kv_linear(cond)
        kv = kv.view(first_dim, -1, 2, self.d_model)
        k, v = kv.unbind(2)

        q = self.q_norm(q).view(first_dim, -1, self.num_heads, self.head_dim)
        k = self.k_norm(k).view(first_dim, -1, self.num_heads, self.head_dim)
        v = v.view(first_dim, -1, self.num_heads, self.head_dim)

        if _xformers_available:
            attn_bias = None
            if mask is not None:
                attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
            x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if mask is not None and mask.ndim == 2:
                mask = (1 - mask.to(x.dtype)) * -10000.0
                mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop.p, is_causal=False)
            x = x.transpose(1, 2)

        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.transpose(1, 2).view(B, C, H, W)

        return x

class SGNet_with_MHCA(nn.Module):
    def __init__(self, num_feats, kernel_size, scale):
        super(SGNet_with_MHCA, self).__init__()
        self.conv_rgb1 = nn.Conv2d(in_channels=3, out_channels=num_feats,
                                   kernel_size=kernel_size, padding=1)
        self.rgb_rb2 = ResBlock(default_conv, num_feats, kernel_size, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1)
        self.rgb_rb3 = ResBlock(default_conv, num_feats, kernel_size, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1)
        self.rgb_rb4 = ResBlock(default_conv, num_feats, kernel_size, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1)

        self.conv_dp1 = nn.Conv2d(in_channels=1, out_channels=num_feats,
                                  kernel_size=kernel_size, padding=1)
        self.conv_dp2 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
                                  kernel_size=kernel_size, padding=1)
        self.dp_rg1 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=16, n_resblocks=6)
        self.dp_rg2 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=16, n_resblocks=6)
        self.dp_rg3 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=16, n_resblocks=6)
        self.dp_rg4 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=16, n_resblocks=6)

        self.bridge1 = SDM(channels=num_feats, rgb_channels=num_feats, scale=scale)
        self.bridge2 = SDM(channels=num_feats, rgb_channels=num_feats, scale=scale)
        self.bridge3 = SDM(channels=num_feats, rgb_channels=num_feats, scale=scale)

        self.c_de = default_conv(num_feats, 2*num_feats, 1)

        my_tail = [
            ResidualGroup(
                default_conv, 3*num_feats, kernel_size, reduction=16, n_resblocks=8),
            ResidualGroup(
                default_conv, 3*num_feats, kernel_size, reduction=16, n_resblocks=8),
            ResidualGroup(
                default_conv, 3*num_feats, kernel_size, reduction=16, n_resblocks=8)
        ]
        self.tail = nn.Sequential(*my_tail)

        self.upsampler = DenseProjection(num_feats, 3*num_feats, scale, up=True, bottleneck=False)
        last_conv = [
            default_conv(3*num_feats, num_feats, kernel_size=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(num_feats, 1, kernel_size=3, bias=True)
        ]
        self.last_conv = nn.Sequential(*last_conv)
        self.bicubic = nn.Upsample(scale_factor=scale, mode='bicubic')

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.gradNet = GCM(n_feats=num_feats, scale=scale)

        self.mhca1 = MultiHeadCrossAttention(d_model=num_feats, num_heads=4)
        self.mhca2 = MultiHeadCrossAttention(d_model=num_feats, num_heads=4)
        self.mhca3 = MultiHeadCrossAttention(d_model=num_feats, num_heads=4)
        self.mhca4 = MultiHeadCrossAttention(d_model=num_feats, num_heads=4)
        self.mhca5 = MultiHeadCrossAttention(d_model=num_feats, num_heads=4)
        self.mhca6 = MultiHeadCrossAttention(d_model=num_feats, num_heads=4)
        self.mhca7 = MultiHeadCrossAttention(d_model=num_feats, num_heads=4)

    def forward(self, x):
        image, depth = x

        out_re, grad_d4 = self.gradNet(depth, image)

        dp_in = self.act(self.conv_dp1(depth))
        dp1 = self.dp_rg1(dp_in)

        dp1_ = self.mhca1(dp1, grad_d4)
        rgb1 = self.act(self.conv_rgb1(image))
        rgb2 = self.rgb_rb2(rgb1)
        ca1_in, r1 = self.bridge1(dp1_, rgb2)

        dp2 = self.dp_rg2(self.mhca2(dp1, ca1_in + dp_in))

        dp2_ = self.mhca3(dp2, grad_d4)
        rgb3 = self.rgb_rb3(r1)
        ca2_in, r2 = self.bridge2(dp2_, rgb3)
        ca2_in_ = ca2_in + self.conv_dp2(dp_in)

        dp3 = self.dp_rg3(self.mhca4(dp2, ca2_in_))
        rgb4 = self.rgb_rb4(r2)

        dp3_ = self.mhca5(dp3, grad_d4)
        ca3_in, r3 = self.bridge3(dp3_, rgb4)
        mhca_all = self.mhca6(self.mhca6(self.mhca6(dp1, dp2), dp3), ca3_in)
        dp4 = self.dp_rg4(mhca_all)

        tail_in = self.upsampler(dp4)
        out = self.last_conv(self.tail(tail_in))

        out = out + self.bicubic(depth)

        return out, out_re
