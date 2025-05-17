import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

class SwinIR_Depth(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3, embed_dim=96,
                 depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6], window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 ape=False, patch_norm=True, use_checkpoint=False, upscale=4,
                 img_range=1., resi_connection='1conv'):
        super(SwinIR_Depth, self).__init__()
        self.upscale = upscale
        self.window_size = window_size
        self.img_range = img_range

        # RGB Encoder (HR processing)
        self.rgb_conv_first = nn.Conv2d(3, embed_dim, 3, 1, 1)
        self.rgb_patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                        in_chans=embed_dim, embed_dim=embed_dim)
        self.rgb_patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size,
                                            in_chans=embed_dim, embed_dim=embed_dim)
        self.rgb_layers = nn.ModuleList()
        for i in range(len(depths)):
            rgb_layer = RSTB(dim=embed_dim,
                           input_resolution=(self.rgb_patch_embed.patches_resolution[0],
                                             self.rgb_patch_embed.patches_resolution[1]),
                           depth=depths[i],
                           num_heads=num_heads[i],
                           window_size=window_size,
                           mlp_ratio=mlp_ratio,
                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                           drop=drop_rate, attn_drop=attn_drop_rate,
                           drop_path=drop_path_rate,
                           norm_layer=norm_layer,
                           img_size=img_size,
                           patch_size=patch_size,
                           resi_connection=resi_connection)
            self.rgb_layers.append(rgb_layer)
        self.rgb_norm = norm_layer(embed_dim)
        self.rgb_conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # Depth Encoder (LR processing with upsampling)
        self.depth_conv_first = nn.Conv2d(1, embed_dim, 3, 1, 1)
        self.depth_patch_embed = PatchEmbed(img_size=img_size//upscale,
                                          patch_size=patch_size,
                                          in_chans=embed_dim,
                                          embed_dim=embed_dim)
        self.depth_patch_unembed = PatchUnEmbed(img_size=img_size//upscale,
                                          patch_size=patch_size,
                                          in_chans=embed_dim,
                                          embed_dim=embed_dim)
        self.depth_layers = nn.ModuleList()
        for i in range(len(depths)):
            depth_layer = RSTB(dim=embed_dim,
                             input_resolution=(self.depth_patch_embed.patches_resolution[0],
                                               self.depth_patch_embed.patches_resolution[1]),
                             depth=depths[i],
                             num_heads=num_heads[i],
                             window_size=window_size,
                             mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                             drop=drop_rate, attn_drop=attn_drop_rate,
                             drop_path=drop_path_rate,
                             norm_layer=norm_layer,
                             img_size=img_size//upscale,
                             patch_size=patch_size,
                             resi_connection=resi_connection)
            self.depth_layers.append(depth_layer)
        self.depth_norm = norm_layer(embed_dim)
        self.depth_conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.depth_upsample = UpsampleOneStep(upscale, embed_dim, embed_dim,
                                            (img_size//upscale, img_size//upscale))

        # Feature Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(2*embed_dim, 2*embed_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(2*embed_dim, embed_dim, 3, 1, 1)
        )

        # Reconstruction
        self.conv_last = nn.Conv2d(embed_dim, 1, 3, 1, 1)

        self.apply(self._init_weights)
        self.mean = torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1) if in_chans==3 else torch.zeros(1, 1, 1, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, encoder_type='rgb'):
        if encoder_type == 'rgb':
            x_size = (x.shape[2], x.shape[3])
            x = self.rgb_patch_embed(x)
            for layer in self.rgb_layers:
                x = layer(x, x_size)
            x = self.rgb_norm(x)
            x = self.rgb_patch_unembed(x, x_size)
        else:
            x_size = (x.shape[2], x.shape[3])
            x = self.depth_patch_embed(x)
            for layer in self.depth_layers:
                x = layer(x, x_size)
            x = self.depth_norm(x)
            x = self.depth_patch_unembed(x, x_size)
        return x

    def forward(self, rgb, depth):
        H, W = rgb.shape[2:]
        rgb = self.check_image_size(rgb)
        depth = self.check_image_size(depth)

        # Normalize
        self.mean = self.mean.to(rgb.device)
        rgb = (rgb - self.mean) * self.img_range

        # RGB Encoder
        rgb_feat = self.rgb_conv_first(rgb)
        rgb_feat = self.rgb_conv_after_body(self.forward_features(rgb_feat, 'rgb')) + rgb_feat

        # Depth Encoder
        depth_feat = self.depth_conv_first(depth)
        depth_feat = self.depth_conv_after_body(self.forward_features(depth_feat, 'depth')) + depth_feat
        depth_feat = self.depth_upsample(depth_feat)

        # Feature Fusion
        fused = torch.cat([rgb_feat, depth_feat], dim=1)
        fused = self.fusion(fused)

        # Reconstruction
        out = self.conv_last(fused)
        out = out / self.img_range + (self.mean.mean() if self.mean.shape[1]==3 else 0)

        return out[:, :, :H*self.upscale, :W*self.upscale]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

def load_pretrained_swinir(model_path, swinir_depth_model):
    pretrained_dict = torch.load(model_path)
    if 'params_ema' in pretrained_dict:
        pretrained_dict = pretrained_dict['params_ema']
    elif 'params' in pretrained_dict:
        pretrained_dict = pretrained_dict['params']

    rgb_encoder_dict = {}
    key_mapping = {
        'conv_first': 'rgb_conv_first',
        'conv_after_body': 'rgb_conv_after_body',
        'patch_embed': 'rgb_patch_embed',
        'patch_unembed': 'rgb_patch_unembed',
        'layers': 'rgb_layers',
        'norm': 'rgb_norm'
    }

    for k, v in pretrained_dict.items():
        if 'conv_last' in k or 'upsample' in k:
            continue
        for old_key, new_key in key_mapping.items():
            if old_key in k:
                new_k = k.replace(old_key, new_key)
                rgb_encoder_dict[new_k] = v
                break
    model_dict = swinir_depth_model.state_dict()
    model_dict.update(rgb_encoder_dict)
    swinir_depth_model.load_state_dict(model_dict, strict=False)

    # Freeze RGB encoder parameters
    for name, param in model.named_parameters():
        if any(x in name for x in ['rgb_conv_first', 'rgb_layers', 'rgb_norm', 'rgb_conv_after_body']):
            param.requires_grad = False

    return swinir_depth_model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    upscale = 4
    window_size = 8
    hr_size = 128
    lr_size = hr_size // upscale

    model = SwinIR_Depth(img_size=hr_size, upscale=upscale, window_size=window_size).to(device)
    rgb_input = torch.randn(1, 3, hr_size, hr_size).to(device)
    depth_input = torch.randn(1, 1, lr_size, lr_size).to(device)
    output = model(rgb_input, depth_input)
    print(output.shape)  # Should be (1, 1, 128, 128)
