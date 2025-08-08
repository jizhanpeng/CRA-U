import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *
__all__ = ['UNext','NetWork']

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
from mmcv.cnn import ConvModule
import pdb

from einops import rearrange


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


def shift(dim):
            x_shift = [ torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, self.pad, H)
            x_cat = torch.narrow(x_cat, 3, self.pad, W)
            return x_cat

class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)


        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_r = x_s.transpose(1,2)


        x = self.fc1(x_shift_r)

        x = self.dwconv(x, H, W)
        x = self.act(x) 
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_c = x_s.transpose(1,2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x

class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class NetWork(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP
    
    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=256, patch_size=16, in_chans=3,  embed_dims=[32, 64, 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )

        ##  左边第一

        self.encoder1 = InvertedResidual(in_channels = 16, out_channels = embed_dims[0], expansion_factor = 2, stride = 1)
        self.ebn1 = nn.BatchNorm2d(embed_dims[0])

        ##  左边第二层

        self.encoder2 = InvertedResidual(in_channels = embed_dims[0], out_channels = embed_dims[1], expansion_factor = 2, stride = 1)
        self.ebn2 = nn.BatchNorm2d(embed_dims[1])

        ##  左边第三层

        self.encoder3 = InvertedResidual(in_channels = embed_dims[1], out_channels = embed_dims[2], expansion_factor = 2, stride = 1)
        self.ebn3 = nn.BatchNorm2d(embed_dims[2])


        ##  左边第四层

        self.encoder4 = InvertedResidual(in_channels = embed_dims[2], out_channels = embed_dims[3], expansion_factor = 2, stride = 1)
        self.ebn4 = nn.BatchNorm2d(embed_dims[3])

        num_heads = [4, 8]

        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=1, stride=1, in_chans=embed_dims[3],
                                              embed_dim=embed_dims[3])


        self.block4 = Block(dim=embed_dims[3], num_patches=img_size*img_size // (16*16), num_heads=num_heads[0], mlp_ratio=4., attn_drop=0., drop_path=0.)

        self.norm4 = nn.BatchNorm2d(embed_dims[3])

        ##  第五层


        self.encoder5 = InvertedResidual(in_channels = embed_dims[3], out_channels = embed_dims[4], expansion_factor = 2, stride = 1)
        self.ebn5 = nn.BatchNorm2d(embed_dims[4])


        self.patch_embed5 = OverlapPatchEmbed(img_size=img_size// 16, patch_size=1, stride=1, in_chans=embed_dims[4],
                                              embed_dim=embed_dims[4])

        self.block5 = Block(dim=embed_dims[4], num_patches=img_size*img_size // (32*32), num_heads=num_heads[1], mlp_ratio=4., attn_drop=0., drop_path=0.)

        self.norm5 = nn.BatchNorm2d(embed_dims[4])


        self.decoder5 = InvertedResidual(in_channels = embed_dims[4], out_channels = embed_dims[3], expansion_factor = 2, stride = 1)
        self.dbn5 = nn.BatchNorm2d(embed_dims[3])


        ##  右边第四层

        # self.dblock4 = shift_Block(dim=embed_dims[1], mlp_ratio=2., attn_drop=0., drop_path=0., pixel=2, step=1,
        #                         step_pad_mode='c', pixel_pad_mode='c', shift_size=5)



        self.patch_dembed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=1, stride=1, in_chans=embed_dims[3],
                                              embed_dim=embed_dims[3])
        self.dblock4 = Block(dim=embed_dims[3], num_patches=img_size*img_size // (16*16), num_heads=num_heads[0], mlp_ratio=4., attn_drop=0.,
                            drop_path=0.)
        self.dnorm4 = nn.BatchNorm2d(embed_dims[3])


        self.decoder4 = InvertedResidual(in_channels = embed_dims[3], out_channels = embed_dims[2], expansion_factor = 2, stride = 1)
        self.dbn4 = nn.BatchNorm2d(embed_dims[2])


        ##  右边第三层

        self.decoder3 = InvertedResidual(in_channels = embed_dims[2], out_channels = embed_dims[1], expansion_factor = 2, stride = 1)
        self.dbn3 = nn.BatchNorm2d(embed_dims[1])


        ##  右边第二层

        self.decoder2 = InvertedResidual(in_channels = embed_dims[1], out_channels = embed_dims[0], expansion_factor = 2, stride = 1)
        self.dbn2 = nn.BatchNorm2d(embed_dims[0])


        ##  右边第一层

        self.decoder1 = InvertedResidual(in_channels = embed_dims[0], out_channels = 16, expansion_factor = 2, stride = 1)
        self.dbn1 = nn.BatchNorm2d(16)


        self.final = nn.Conv2d(16, num_classes, kernel_size=1)


        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        
        B = x.shape[0]

        x = self.in_conv(x)

        ##  左边第一层
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out

        ##  左边第二层
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out

        ##  左边第三层

        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        ##  左边第四层

        out = F.relu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))

        out,H,W = self.patch_embed4(out)
        out = self.block4(out,H,W)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.norm4(out)

        t4 = out

        ### 第五层

        out = F.relu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))

        out ,H,W= self.patch_embed5(out)
        out = self.block5(out,H,W)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.norm5(out)

        out = F.relu(F.interpolate(self.dbn5(self.decoder5(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t4)

        ##  右边第四层

        _,_,H,W = out.shape

        out ,H,W= self.patch_dembed4(out)
        out = self.dblock4(out,H,W)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.dnorm4(out)

        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)

        ##  右边第三层

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)

        ##  右边第二层

        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)

        ##  右边第一层

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))

        return self.final(out)

#EOF


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Tok_MLP(nn.Module):
    def __init__(self, dim, attn_drop=0., proj_drop=0., pixel=2,
                 step=1, step_pad_mode='c', pixel_pad_mode='c', shift_size=5):
        super().__init__()
        """
        self.pixel: h and w in inner-region rearrangement
        self.step: s in cross-region rearrangement
        """
        self.pixel = pixel
        self.step = step
        self.step_pad_mode = step_pad_mode
        self.pixel_pad_mode = pixel_pad_mode
        print('pixel: {} pad mode: {} step: {} pad mode: {}'.format(
            pixel, pixel_pad_mode, step, step_pad_mode))

        self.mlp_h1 = nn.Conv2d(dim, dim , 1, bias=False)
        self.mlp_h1_norm = nn.BatchNorm2d(dim)
        self.mlp_h2 = nn.Conv2d(dim, dim , 1, bias=True)
        self.mlp_w1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.mlp_w1_norm = nn.BatchNorm2d(dim)
        self.mlp_w2 = nn.Conv2d(dim, dim, 1, bias=True)
        self.mlp_c = nn.Conv2d(dim, dim, 1, bias=True)

        self.act = nn.ReLU()

        self.shift_size = shift_size
        self.pad = 2

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        h: H x W x C -> H/pixel x W x C*pixel
        w: H x W x C -> H x W/pixel x C*pixel
        Setting of F.pad: (left, right, top, bottom)
        """
        B, C, H, W = x.shape

        h, w = x.clone(), x.clone()

        # h = F.pad(h, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        h = torch.chunk(h, self.shift_size, 1)
        h = [torch.roll(x_c, shift, 2) for x_c, shift in zip(h, range(-self.pad, self.pad + 1))]
        h = torch.cat(h, 1)
        # x_cat = torch.narrow(x_cat, 2, self.pad, H)
        # x_s = torch.narrow(x_cat, 3, self.pad, W)

        h = self.mlp_h1(h)
        h = self.mlp_h1_norm(h)
        h = self.act(h)
        h = self.mlp_h2(h)

        h = torch.chunk(h, self.shift_size, 1)
        h = [torch.roll(x_c, -shift, 2) for x_c, shift in zip(h, range(-self.pad, self.pad + 1))]
        h = torch.cat(h, 1)

        # w = F.pad(h, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        w = torch.chunk(w, self.shift_size, 1)
        w = [torch.roll(x_c, shift, 3) for x_c, shift in zip(w, range(-self.pad, self.pad + 1))]
        w = torch.cat(w, 1)
        # x_cat = torch.narrow(x_cat, 2, self.pad, H)
        # x_s = torch.narrow(x_cat, 3, self.pad, W)

        w = self.mlp_w1(w)
        w = self.mlp_w1_norm(w)
        w = self.act(w)
        w = self.mlp_w2(w)

        w = torch.chunk(w, self.shift_size, 1)
        w = [torch.roll(x_c, -shift, 3) for x_c, shift in zip(w, range(-self.pad, self.pad + 1))]
        w = torch.cat(w, 1)

        c = self.mlp_c(x)

        a = (h + w + c).flatten(2).mean(2).unsqueeze(2).unsqueeze(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(3).unsqueeze(3)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class shift_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 pixel=2, step=1, step_pad_mode='c', pixel_pad_mode='c', shift_size=5):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Tok_MLP(
            dim, attn_drop=attn_drop, pixel=pixel, step=step,
            step_pad_mode=step_pad_mode, pixel_pad_mode=pixel_pad_mode, shift_size=5)

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class Conv(nn.Module):
    def __init__(self, in_channels , out_channels ):
        super(Conv, self).__init__()
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, stride=1, groups=in_channels, padding_mode='reflect') # depthwise conv
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.pwconv1 = nn.Linear(in_channels, in_channels)  # pointwise/1x1 convs, implemented with linear layers
        self.act1 = nn.GELU()
        self.pwconv2 = nn.Linear(in_channels, out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()
    def forward(self, x):
        x = self.dwconv(x)
        x = self.norm1(x)
        residual = x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.norm2(x)
        x = self.act2(residual + x)

        return x



# DW卷积
def Conv3x3BNReLU(in_channels,out_channels,stride,groups):
    return nn.Sequential(
            # stride=2 wh减半，stride=1 wh不变
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=stride, padding=3, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

# PW卷积
def Conv1x1BNReLU(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

# # PW卷积(Linear) 没有使用激活函数
def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

class InvertedResidual(nn.Module):
    # t = expansion_factor,也就是扩展因子，文章中取6
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = (in_channels * expansion_factor)
        # print("expansion_factor:", expansion_factor)
        # print("mid_channels:",mid_channels)

        # 先1x1卷积升维，再1x1卷积降维
        self.bottleneck = nn.Sequential(
            # 升维操作: 扩充维度是 in_channels * expansion_factor (6倍)
            Conv1x1BNReLU(in_channels, mid_channels),
            # DW卷积,降低参数量
            Conv3x3BNReLU(mid_channels, mid_channels, stride, groups=mid_channels),
            # 降维操作: 降维度 in_channels * expansion_factor(6倍) 降维到指定 out_channels 维度
            Conv1x1BN(mid_channels, out_channels)
        )

        # 第一种: stride=1 才有shortcut 此方法让原本不相同的channels相同
        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.bottleneck(x)
        # 第一种:
        out = (out+self.shortcut(x)) if self.stride==1 else out
        return out



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class FocusedLinearAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 focusing_factor=3, kernel_size=5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))
        print('Linear Attention sr_ratio{} f{} kernel{}'.
              format(sr_ratio, focusing_factor, kernel_size))

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        k = k + self.positional_encoding

        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])

        # focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        # scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6


        # q = q / scale
        # k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        # q = q ** focusing_factor
        # k = k ** focusing_factor


        q_1 = torch.argsort(q, dim=2)
        k_1 = torch.argsort(k, dim=2)
        q = q * q_1
        k = k * k_1


        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm




        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.permute(0, 2, 1), size=x.shape[1], mode='linear').permute(0, 2, 1)
        num = int(v.shape[1] ** 0.5)
        feature_map = rearrange(v, "b (w h) c -> b c w h", w=num, h=num)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")
        x = x + feature_map
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_patches, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1,
                 focusing_factor=3, kernel_size=5, attn_type='L'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        assert attn_type in ['L', 'S']
        if attn_type == 'L':
            self.attn = FocusedLinearAttention(
                dim, num_patches,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,
                focusing_factor=focusing_factor, kernel_size=kernel_size)
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x