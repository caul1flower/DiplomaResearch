import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from sgnet.sdm import SDB, stdv_channels, DenseBlock, InvBlock
from swin2sr import PatchEmbed, PatchUnEmbed, RSTB, UpsampleOneStep

class SDM(nn.Module):
    def __init__(self, channels, rgb_channels):
        super(SDM, self).__init__()
        self.rgbprocess = nn.Conv2d(rgb_channels, rgb_channels, 3, 1, 1)
        self.rgbpre = nn.Conv2d(rgb_channels, rgb_channels, 1, 1, 0)
        self.spa_process = nn.Sequential(InvBlock(DenseBlock, channels + rgb_channels, channels),
                                         nn.Conv2d(channels + rgb_channels, channels, 1, 1, 0))
        self.fre_process = SDB(channels, rgb_channels)
        self.spa_att = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.cha_att = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=1, padding=0, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, kernel_size=1, padding=0, bias=True),
                                     nn.Sigmoid())
        self.post = nn.Conv2d(channels, channels, 3, 1, 1)

        self.fuse_process = nn.Sequential(InvBlock(DenseBlock, 2*channels, channels),
                                         nn.Conv2d(2*channels, channels, 1, 1, 0))

    def forward(self, dp, rgb):  # , i
        rgbpre = self.rgbprocess(rgb)
        rgb = self.rgbpre(rgbpre)

        spafuse = self.spa_process(torch.cat([dp, rgb], 1))
        frefuse = self.fre_process(dp, rgb)

        cat_f = torch.cat([spafuse, frefuse], 1)
        cat_f = self.fuse_process(cat_f)

        cha_res = self.cha_att(self.contrast(cat_f) + self.avgpool(cat_f)) * cat_f
        out = cha_res + dp

        return out



class SGTNet(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3, embed_dim=96,
                 depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6], window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 ape=False, patch_norm=True, use_checkpoint=False, upscale=4,
                 img_range=1., resi_connection='1conv'):
        super(SGTNet, self).__init__()
        self.upscale = upscale
        self.window_size = window_size
        self.img_range = img_range
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
                           qkv_bias=qkv_bias,
                           drop=drop_rate, attn_drop=attn_drop_rate,
                           drop_path=drop_path_rate,
                           norm_layer=norm_layer,
                           img_size=img_size,
                           patch_size=patch_size,
                           resi_connection=resi_connection)
            self.rgb_layers.append(rgb_layer)
        self.rgb_norm = norm_layer(embed_dim)
        self.rgb_conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)


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
                             qkv_bias=qkv_bias,
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

        self.freq_fusion = SDM(channels=embed_dim, rgb_channels=embed_dim)

        self.fusion = nn.Sequential(
            nn.Conv2d(3 * embed_dim, 2 * embed_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * embed_dim, embed_dim, 3, 1, 1)
        )

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
        # Frequency Domain Fusion
        freq_fused = self.freq_fusion(depth_feat, rgb_feat)

        # Combine features
        fused = torch.cat([rgb_feat, depth_feat, freq_fused], dim=1)
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


def load_pretrained_swin2sr(pretrained_path, model, encoder_type='rgb'):
    print(f"Loading pretrained model from {pretrained_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(pretrained_path):
        print(f"Error: Pretrained model file not found at {pretrained_path}")
        return model

    try:
        pretrained_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
        if 'params' in pretrained_dict:
            pretrained_dict = pretrained_dict['params']
        elif 'model' in pretrained_dict:
            pretrained_dict = pretrained_dict['model']
        elif 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        elif 'params_ema' in pretrained_dict:
            pretrained_dict = pretrained_dict['params_ema']

        model_dict = model.state_dict()
        rgb_encoder_dict = {}

        key_mapping = {
            'conv_first': f'{encoder_type}_conv_first',
            'conv_after_body': f'{encoder_type}_conv_after_body',
            'patch_embed': f'{encoder_type}_patch_embed',
            'patch_unembed': f'{encoder_type}_patch_unembed',
            'layers': f'{encoder_type}_layers',
            'norm': f'{encoder_type}_norm'
        }

        loaded_params = 0
        skipped_params = 0
        for k, v in pretrained_dict.items():
            if 'conv_last' in k or 'upsample' in k:
                continue
            mapped = False
            for old_key, new_key in key_mapping.items():
                if old_key in k:
                    new_k = k.replace(old_key, new_key)
                    if new_k in model_dict and model_dict[new_k].shape == v.shape:
                        rgb_encoder_dict[new_k] = v
                        loaded_params += 1
                    else:
                        if new_k in model_dict:
                            print(f"Shape mismatch for {new_k}: {v.shape} vs {model_dict[new_k].shape}")
                        skipped_params += 1
                    mapped = True
                    break

            if not mapped:
                skipped_params += 1

        model_dict.update(rgb_encoder_dict)
        model.load_state_dict(model_dict, strict=False)
        frozen = 0
        for name, param in model.named_parameters():
            if f'{encoder_type}_' in name:
                param.requires_grad = False
                frozen += 1

        print(f"Successfully loaded {loaded_params} parameters, skipped {skipped_params} parameters, frozen {frozen} parameters")

    except Exception as e:
        print(f"Error loading pretrained model: {e}")

    return model
