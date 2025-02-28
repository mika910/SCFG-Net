import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from networks.segformer import *
from networks.merit_lib.networks import MaxViT4Out_Small
from networks.MLKA import MLKA
from networks.SCFG_block import SCFG
from timm.models.layers import DropPath, to_2tuple
import math


class DWConvLKA(nn.Module):
    def __init__(self, dim=768):
        super(DWConvLKA, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConvLKA(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class ChannelAttention(nn.Module):
    """
    Input -> x: [B, N, C]
    Output -> [B, N, C]
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(
            x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2)
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())

        return x


class MyDecoderLayerMLKA(nn.Module):
    def __init__(
            self, input_size, in_out_chan,  head_count,token_mlp_mode, reduction_ratio, n_class=9,
            norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        reduction_ratio = reduction_ratio
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.scfg_attn = SCFG(inp=out_dim, out=out_dim)
            self.scfg_attn_norm = nn.LayerNorm(out_dim)
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.scfg_attn = SCFG(inp=out_dim, out=out_dim)
            self.scfg_attn_norm = nn.LayerNorm(out_dim)
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)
        self.layer_mlka_1 = MLKA(out_dim)
        self.layer_mlka_2 = MLKA(out_dim)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            x2 = x2.contiguous()
            # b, c, h, w = x1.shape
            b2, h2, w2, c2 = x2.shape
            x2 = x2.view(b2, -1, c2)
            x1_expand = self.x1_linear(x1)
            x2_new = x2.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2)
            x1_expand = x1_expand.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2)
            attn_gate = self.scfg_attn(x2_new, x1_expand)  # B C H W
            cat_linear_x = x1_expand + attn_gate  # B C H W
            cat_linear_x = cat_linear_x.permute(0, 2, 3, 1)  # B H W C
            cat_linear_x = self.scfg_attn_norm(cat_linear_x)  # B H W C
            cat_linear_x = cat_linear_x.permute(0, 3, 1, 2).contiguous()  # B C H W
            tran_layer_1 = self.layer_mlka_1(cat_linear_x)
            tran_layer_2 = self.layer_mlka_2(tran_layer_1)
            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2),
                                             tran_layer_2.size(1))
            if self.last_layer:
                out = self.last_layer(
                    self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2))  # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2)  # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out


class SCFG_Net(nn.Module):
    def __init__(self, num_classes=None, token_mlp_mode="mix_skip"):
        super().__init__()
        # Encoder
        self.backbone = MaxViT4Out_Small(n_class=num_classes, img_size=224)
        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [96, 96, 96, 96, 96],
            [192, 192, 192, 192, 192],
            [384, 384, 384, 384, 384],
            [768, 768, 768, 768, 768],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]
        reduction_ratio = [16, 8, 6, 2]
        head_count = [32, 16, 1, 1]
        self.decoder_3 = MyDecoderLayerMLKA(
            (d_base_feat_size, d_base_feat_size),
            in_out_chan[3],
            head_count[0],
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[0])
        self.decoder_2 = MyDecoderLayerMLKA(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            head_count[1],
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[1])
        self.decoder_1 = MyDecoderLayerMLKA(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            head_count[2],
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[2])
        self.decoder_0 = MyDecoderLayerMLKA(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            head_count[3],
            token_mlp_mode,
            n_class=num_classes,
            is_last=True,
            reduction_ratio=reduction_ratio[3])

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        output_enc_3, output_enc_2, output_enc_1, output_enc_0 = self.backbone(x)
        b, c, _, _ = output_enc_3.shape
        tmp_3 = self.decoder_3(output_enc_3.permute(0, 2, 3, 1).view(b, -1, c))
        tmp_2 = self.decoder_2(tmp_3, output_enc_2.permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, output_enc_1.permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc_0.permute(0, 2, 3, 1))

        return tmp_0



