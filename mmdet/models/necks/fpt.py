import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from functools import reduce
import operator
from ..builder import NECKS
from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn import ConvModule

def XavierFill(tensor):
    """Caffe2 XavierFill Implementation"""
    size = reduce(operator.mul, tensor.shape, 1)
    fan_in = size / tensor.shape[0]
    scale = math.sqrt(3 / fan_in)
    return init.uniform_(tensor, -scale, scale)


def MSRAFill(tensor):
    """Caffe2 MSRAFill Implementation"""
    size = reduce(operator.mul, tensor.shape, 1)
    fan_out = size / tensor.shape[1]
    scale = math.sqrt(2 / fan_out)
    return init.normal_(tensor, 0, scale)

@NECKS.register_module()
class FPT(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 P2only=False,
                 fpt_rendering=True,
                 USE_GN=True,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')
                 ):
        super(FPT, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.in_channels.reverse()
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.dim_out = out_channels
        fpt_dim = self.dim_out//8
        self.USE_GN = USE_GN
        self.P2only = P2only
        self.fpt_rendering = fpt_rendering
        self.st = SelfTrans(n_head=1, n_mix=4, d_model=fpt_dim, d_k=fpt_dim, d_v=fpt_dim)
        self.rt = RenderTrans(channels_high=fpt_dim, channels_low=fpt_dim, upsample=False)

        min_level, max_level = start_level, end_level
        self.num_backbone_stages = len(in_channels)
        # self.spatial_scale = []

        self.conv_top = nn.Conv2d(self.in_channels[0], fpt_dim, 1, 1, 0)
        if USE_GN:
            self.conv_top = nn.Sequential(
                nn.Conv2d(self.in_channels[0], fpt_dim, 1, 1, 0, bias=False),
                nn.GroupNorm(get_group_gn(fpt_dim), fpt_dim, eps=1e-5))
        else:
            self.conv_top = nn.Conv2d(self.in_channels[0], fpt_dim, 1, 1, 0)

        self.ground_lateral_modules = nn.ModuleList()
        self.posthoc_modules = nn.ModuleList()

        for i in range(self.num_backbone_stages - 1):
            self.ground_lateral_modules.append(
                ground_lateral_module(fpt_dim, self.in_channels[i + 1], fpt_dim, USE_GN=self.USE_GN)
            )

        for i in range(self.num_backbone_stages):
            if USE_GN:
                self.posthoc_modules.append(nn.Sequential(
                    nn.Conv2d(fpt_dim, self.dim_out, 3, 1, 1, bias=False),
                    nn.GroupNorm(get_group_gn(self.dim_out), self.dim_out,
                                 eps=1e-5),
                    # nn.Conv2d(fpt_dim, fpt_dim, 3, 1, 1, bias=False),
                    nn.ReLU(inplace=True)
                ))
            else:
                self.posthoc_modules.append(
                    nn.Conv2d(fpt_dim, self.dim_out, 3, 1, 1, bias=False),
                    # nn.Conv2d(fpt_dim, fpt_dim, 3, 1, 1, bias=False),
                    nn.ReLU(inplace=True)
                )

            # self.spatial_scale.append(fpt_level_info.spatial_scales[i])

        if self.fpt_rendering:
            self.fpt_rendering_conv1_modules = nn.ModuleList()
            self.fpt_rendering_conv2_modules = nn.ModuleList()

            for i in range(self.num_backbone_stages - 1):
                if USE_GN:
                    self.fpt_rendering_conv1_modules.append(nn.Sequential(
                        nn.Conv2d(self.dim_out, fpt_dim, 3, 2, 1, bias=True),
                        nn.GroupNorm(get_group_gn(fpt_dim), fpt_dim,
                                     eps=1e-5), nn.ReLU(inplace=True)
                    ))
                    self.fpt_rendering_conv2_modules.append(nn.Sequential(
                        nn.Conv2d(fpt_dim, self.dim_out, 3, 1, 1, bias=True),
                        nn.GroupNorm(get_group_gn(self.dim_out), self.dim_out, eps=1e-5),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    self.fpt_rendering_conv1_modules.append(
                        nn.Conv2d(self.dim_out, fpt_dim, 3, 2, 1)
                    )
                    self.fpt_rendering_conv2_modules.append(
                        nn.Conv2d(fpt_dim, self.dim_out, 3, 1, 1))

        if not add_extra_convs and max_level == 6:
            self.maxpool_p6 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)
            # self.spatial_scale.insert(0, self.spatial_scale[0] * 0.5)

        if add_extra_convs and max_level > 5:
            self.extra_pyramid_modules = nn.ModuleList()
            dim_in = self.in_channels[0]
            for i in range(6, max_level + 1):
                self.extra_pyramid_modules(
                    nn.Conv2d(dim_in, self.dim_out, 3, 2, 1)
                )
                dim_in = self.dim_out
                # self.spatial_scale.insert(0, self.spatial_scale[0] * 0.5)

        # if self.P2only:
        #     self.spatial_scale = self.spatial_scale[-1]

        self._init_weights()

    def _init_weights(self):
        def init_func(m):
            if isinstance(m, nn.Conv2d):
                XavierFill(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        for child_m in self.children():
            if (not isinstance(child_m, nn.ModuleList) or
                    not isinstance(child_m[0], ground_lateral_module)):
                child_m.apply(init_func)

    def forward(self, x):
        fpt_inner_blobs = [self.st(self.conv_top(x[-1]))]

        for i in range(self.num_backbone_stages - 1):
            fpt_inner_blobs.append(
                self.ground_lateral_modules[i](fpt_inner_blobs[-1], x[-(i + 2)])
            )
        fpt_output_blobs = []

        if self.fpt_rendering:
            fpt_middle_blobs = []

        for i in range(self.num_backbone_stages):
            if not self.fpt_rendering:
                fpt_output_blobs.append(
                    self.posthoc_modules[i](fpt_inner_blobs[i])
                )
            else:
                fpt_middle_blobs.append(
                    self.posthoc_modules[i](fpt_inner_blobs[i])
                )

        if self.fpt_rendering:
            fpt_output_blobs.append(fpt_middle_blobs[-1])
            for i in range(2, self.num_backbone_stages + 1):
                rend_tmp = self.fpt_rendering_conv1_modules[i - 2](fpt_output_blobs[0])
                print(fpt_middle_blobs[self.num_backbone_stages - i].size())
                rend_tmp = rend_tmp + fpt_middle_blobs[self.num_backbone_stages - i]
                # rend_tmp = self.rt(fpt_middle_blobs[self.num_backbone_stages - i], rend_tmp)
                rend_tmp = self.fpt_rendering_conv2_modules[i - 2](rend_tmp)
                fpt_output_blobs.insert(0, rend_tmp)

        if hasattr(self, 'maxpool_p6'):
            fpt_output_blobs.insert(0, self.maxpool_p6(fpt_output_blobs[0]))

        if hasattr(self, 'extra_pyramid_modules'):
            blob_in = x[-1]
            fpt_output_blobs.insert(0, self.extra_pyramid_modules(blob_in))
            for module in self.extra_pyramid_modules[1:]:
                fpt_output_blobs.insert(0, module(F.relu(fpt_output_blobs[0], inplace=True)))

        if self.P2only:
            return fpt_output_blobs[-1]
        else:
            # fpt_output_blobs.reverse()
            return fpt_output_blobs


class ground_lateral_module(nn.Module):
    def __init__(self, dim_in_top, dim_in_lateral, fpn_dim, USE_GN=True, ZERO_INIT_LATERAL=False):
        super().__init__()
        self.USE_GN = USE_GN
        self.ZERO_INIT_LATERAL = ZERO_INIT_LATERAL
        self.dim_in_top = dim_in_top
        self.dim_in_lateral = dim_in_lateral
        self.dim_out = dim_in_top
        if self.USE_GN:
            self.conv_lateral = nn.Sequential(
                nn.Conv2d(dim_in_lateral, self.dim_out, 3, 1, 1, bias=False),
                nn.GroupNorm(get_group_gn(self.dim_out), self.dim_out,
                             eps=1e-5),
                # nn.Conv2d(dim_in_lateral, self.dim_out, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv_lateral = nn.Sequential(
                nn.Conv2d(dim_in_lateral, self.dim_out, 3, 1, 1, bias=False),
                # nn.Conv2d(dim_in_lateral, self.dim_out, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True)
            )

        self._init_weights()
        self.st = SelfTrans(n_head=1, n_mix=4, d_model=fpn_dim, d_k=fpn_dim, d_v=fpn_dim)
        self.gt = GroundTrans(in_channels=fpn_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)

    def _init_weights(self):
        if self.USE_GN:
            conv = self.conv_lateral[0]
        else:
            conv = self.conv_lateral

        if self.ZERO_INIT_LATERAL:
            init.constant_(conv.weight, 0)
        else:
            XavierFill(conv.weight)
        if conv.bias is not None:
            init.constant_(conv.bias, 0)

    def forward(self, top_blob, lateral_blob):
        lat = self.conv_lateral(lateral_blob)
        lat = self.st(lat)
        # td = top_blob
        return self.gt(top_blob, lat)


# def get_min_max_levels():
#     min_level = 2
#     max_level = 5
#     if cfg.FPN.MULTILEVEL_RPN and not cfg.FPN.MULTILEVEL_ROIS:
#         max_level = cfg.FPN.RPN_MAX_LEVEL
#         min_level = cfg.FPN.RPN_MIN_LEVEL
#     if not cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
#         max_level = cfg.FPN.ROI_MAX_LEVEL
#         min_level = cfg.FPN.ROI_MIN_LEVEL
#     if cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
#         max_level = max(cfg.FPN.RPN_MAX_LEVEL, cfg.FPN.ROI_MAX_LEVEL)
#         min_level = min(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.ROI_MIN_LEVEL)
#     return min_level, max_level


BatchNorm2d = nn.BatchNorm2d


class SelfTrans(nn.Module):
    def __init__(self, n_head, n_mix, d_model, d_k, d_v,
                 norm_layer=BatchNorm2d, kq_transform='conv', value_transform='conv',
                 pooling=True, concat=False, dropout=0.1):
        super(SelfTrans, self).__init__()

        self.n_head = n_head
        self.n_mix = n_mix
        self.d_k = d_k
        self.d_v = d_v

        self.pooling = pooling
        self.concat = concat

        if self.pooling:
            self.pool = nn.AvgPool2d(3, 2, 1, count_include_pad=False)
        if kq_transform == 'conv':
            self.conv_qs = nn.Conv2d(d_model, n_head * d_k, 1)
            nn.init.normal_(self.conv_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        elif kq_transform == 'ffn':
            self.conv_qs = nn.Sequential(
                nn.Conv2d(d_model, n_head * d_k, 3, padding=1, bias=False),
                norm_layer(n_head * d_k),
                nn.ReLU(True),
                nn.Conv2d(n_head * d_k, n_head * d_k, 1),
            )
            nn.init.normal_(self.conv_qs[-1].weight, mean=0, std=np.sqrt(1.0 / d_k))
        elif kq_transform == 'dffn':
            self.conv_qs = nn.Sequential(
                nn.Conv2d(d_model, n_head * d_k, 3, padding=4, dilation=4, bias=False),
                norm_layer(n_head * d_k),
                nn.ReLU(True),
                nn.Conv2d(n_head * d_k, n_head * d_k, 1),
            )
            nn.init.normal_(self.conv_qs[-1].weight, mean=0, std=np.sqrt(1.0 / d_k))
        else:
            raise NotImplemented

        self.conv_ks = self.conv_qs
        if value_transform == 'conv':
            self.conv_vs = nn.Conv2d(d_model, n_head * d_v, 1)
        else:
            raise NotImplemented

        nn.init.normal_(self.conv_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = MixtureOfSoftMax(n_mix=n_mix, d_k=d_k)

        self.conv = nn.Conv2d(n_head * d_v, d_model, 1, bias=False)
        self.norm_layer = norm_layer(d_model)

    def forward(self, x):
        residual = x
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        b_, c_, h_, w_ = x.size()
        if self.pooling:
            qt = self.conv_ks(x).view(b_ * n_head, d_k, h_ * w_)
            kt = self.conv_ks(self.pool(x)).view(b_ * n_head, d_k, h_ * w_ // 4)
            vt = self.conv_vs(self.pool(x)).view(b_ * n_head, d_v, h_ * w_ // 4)
        else:
            kt = self.conv_ks(x).view(b_ * n_head, d_k, h_ * w_)
            qt = kt
            vt = self.conv_vs(x).view(b_ * n_head, d_v, h_ * w_)

        output, attn = self.attention(qt, kt, vt)

        output = output.transpose(1, 2).contiguous().view(b_, n_head * d_v, h_, w_)

        output = self.conv(output)
        if self.concat:
            output = torch.cat((self.norm_layer(output), residual), 1)
        else:
            output = self.norm_layer(output) + residual
        return output


class MixtureOfSoftMax(nn.Module):
    """"https://arxiv.org/pdf/1711.03953.pdf"""

    def __init__(self, n_mix, d_k, attn_dropout=0.1):
        super(MixtureOfSoftMax, self).__init__()
        self.temperature = np.power(d_k, 0.5)
        self.n_mix = n_mix
        self.att_drop = attn_dropout
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.d_k = d_k
        if n_mix > 1:
            self.weight = nn.Parameter(torch.Tensor(n_mix, d_k))
            std = np.power(n_mix, -0.5)
            self.weight.data.uniform_(-std, std)

    def forward(self, qt, kt, vt):
        B, d_k, N = qt.size()
        m = self.n_mix
        assert d_k == self.d_k
        d = d_k // m
        if m > 1:
            bar_qt = torch.mean(qt, 2, True)
            pi = self.softmax1(torch.matmul(self.weight, bar_qt)).view(B * m, 1, 1)

        q = qt.view(B * m, d, N).transpose(1, 2)
        N2 = kt.size(2)
        kt = kt.view(B * m, d, N2)
        v = vt.transpose(1, 2)
        attn = torch.bmm(q, kt)
        attn = attn / self.temperature
        attn = self.softmax2(attn)
        attn = self.dropout(attn)
        if m > 1:
            attn = (attn * pi).view(B, m, N, N2).sum(1)
        output = torch.bmm(attn, v)
        return output, attn

class RenderTrans(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(RenderTrans, self).__init__()
        self.upsample = upsample

        self.conv3x3 = nn.Conv2d(channels_high, channels_high, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_high)

        self.conv1x1 = nn.Conv2d(channels_low, channels_high, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_high)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_low, channels_high, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_high)
        else:
            self.conv_reduction = nn.Conv2d(channels_low, channels_high, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_high)

        self.str_conv3x3 = nn.Conv2d(channels_low, channels_high, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Conv2d(channels_high*2, channels_high, kernel_size=1, padding=0, bias=False)

    def forward(self, x_high, x_low):
        b, c, h, w = x_low.shape
        x_low_gp = nn.AvgPool2d(x_low.shape[2:])(x_low).view(len(x_low), c, 1, 1)
        x_low_gp = self.conv1x1(x_low_gp)
        x_low_gp = self.bn_low(x_low_gp)
        x_low_gp = self.relu(x_low_gp)

        x_high_mask = self.conv3x3(x_high)
        x_high_mask = self.bn_high(x_high_mask)

        x_att = x_high_mask * x_low_gp
        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.str_conv3x3(x_low)) + x_att)
                # self.conv_cat(torch.cat([self.bn_upsample(self.str_conv3x3(x_low)), x_att], dim=1))
        else:
            out = self.relu(
                self.bn_reduction(self.str_conv3x3(x_low)) + x_att)
                # # self.conv_cat(torch.cat([self.bn_reduction(self.str_conv3x3(x_low)), x_att], dim=1))
        return out


class GroundTrans(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='dot', dimension=2, bn_layer=True):
        super(GroundTrans, self).__init__()
        assert dimension in [1, 2, 3]
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d

        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d

        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x_low, x_high):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """
        batch_size = x_low.size(0)
        g_x = self.g(x_high).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x_low.view(batch_size, self.in_channels, -1)
            phi_x = x_high.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x_low).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x_high).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x_low).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x_high).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N
        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0, 2, 1).contiguous()
        y = y.view((batch_size, self.inter_channels)+ x_low.size()[2:])

        z = self.W_z(y)
        return z


class GroundTrans_lite(nn.Module):
    def __init__(self, d_model, dim_feedforward, nhead=4, dropout=0.1):
        super(GroundTrans_lite, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x_top, x_lateral,
                src_mask = None,
                src_key_padding_mask = None):
        """

        :param x_top: (B, C, H, W)
        :param x_lateral: (B, C, 2*H, 2*W)
        :return: out: (B, C, 2*H, 2*W)
        """
        bs, c, h, w = x_lateral.shape
        x_lateral = x_lateral.flatten(2).permute(2, 0, 1) # (4HW, B, C)
        x_top = x_top.flatten(2).permute(2, 0, 1)
        x_top = self.norm1(x_top)
        x_lateral = self.norm1(x_lateral)
        x_lateral_2 = self.self_attn(x_lateral, x_top, value=x_top, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        x_lateral = x_lateral_2 + self.dropout1(x_lateral_2)
        x_lateral_2 = self.norm2(x_lateral)
        x_lateral_2 = self.linear2(self.dropout(self.activation(self.linear1(x_lateral_2))))
        x_lateral = x_lateral_2 + self.dropout2(x_lateral_2)
        return x_lateral.permute(1, 2, 0).view(bs, c, h, w).contiguous()

def get_group_gn(dim):
    """
    get number of groups used by GroupNorm, based on number of channels
    """
    dim_per_gp = -1
    num_groups = 32

    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0
        group_gn = num_groups
    return group_gn


@NECKS.register_module()
class FPT_lite(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FPT_lite, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.gts = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            if i > 0:
                gt_module = GroundTrans_lite(d_model=out_channels, dim_feedforward=out_channels, nhead=4)
                self.gts.append(gt_module)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # transformer top-down
        for i in range(len(self.gts)-1, 0, -1):
            laterals[i] = self.gts[i](laterals[i+1], laterals[i])

        # build top-down path, transformer can be used here
        used_backbone_levels = len(laterals)
        """
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
        """

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # transformer back
        # for i in range(len(self.gts)-1, 0, -1):
        #     outs[i] = self.gts[i](outs[i+1], outs[i])

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)