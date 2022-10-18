# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
from torch.nn.modules.module import Module


class Embedder(nn.Module):
    def __init__(self, nch, n_freq):
        super().__init__()
        self.nch = nch
        self.n_freq = n_freq
        self.out_ch = nch
        self.freq_fn = [(1, lambda x: x)]
        for i in range(n_freq):
            for fn in [torch.sin, torch.cos]:
                self.freq_fn.append((2 ** i, fn))
                self.out_ch += nch

    def forward(self, x):
        out = torch.cat([fn(x * freq) for freq, fn in self.freq_fn], dim=-1)
        assert out.shape[-1] == self.out_ch
        return out


class MLP(nn.Module):
    def __init__(self, nchs, skips, act, w_norm, act_last, w_norm_last, init_zero_last=False):
        super().__init__()
        self.nchs = copy.deepcopy(nchs)
        self.skips = copy.deepcopy(skips)
        self.mlp = nn.ModuleList()
        for i in range(len(nchs) - 1):
            in_ch = nchs[i] if i not in skips else nchs[i] + nchs[0]
            out_ch = nchs[i + 1]
            if i < len(nchs) - 2:
                layer = nn.utils.weight_norm(nn.Linear(in_ch, out_ch)) if w_norm else nn.Linear(in_ch, out_ch)
            else:
                assert i == len(nchs) - 2
                layer = nn.utils.weight_norm(nn.Linear(in_ch, out_ch)) if w_norm_last else nn.Linear(in_ch, out_ch)
                if init_zero_last:
                    torch.nn.init.zeros_(layer.weight)
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)
            self.mlp.append(layer)

        if act == 'softplus':
            self.act = nn.Softplus(beta=100, threshold=20)
        elif act == 'linear':
            self.act = nn.Identity()
        else:
            raise NotImplementedError('Not implement activation type \'%s\'!' % act)

        if act_last == 'softplus':
            self.act_last = nn.Softplus(beta=100, threshold=20)
        elif act_last == 'linear':
            self.act_last = nn.Identity()
        else:
            raise NotImplementedError('Not implement activation type \'%s\'!' % act_last)

    def forward(self, x):
        x_ = x
        for i in range(len(self.mlp)):
            if i in self.skips:
                x_ = torch.cat([x_, x], dim=-1)
            x_ = self.mlp[i](x_)
            x_ = self.act(x_) if i < len(self.mlp) - 1 else self.act_last(x_)
        return x_


class Conv2dBias(nn.Conv2d):
    def __init__(self, in_ch, out_ch, kernel_size, size, stride, padding, use_bias=True, *args, **kwargs):
        super().__init__(in_ch, out_ch, bias=False, kernel_size=kernel_size, stride=stride, padding=padding, *args, **kwargs)
        self.use_bias = use_bias
        if self.use_bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(1, out_ch, size, size), requires_grad=True))

    def forward(self, x):
        out = F.conv2d(x, self.weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) 
        if self.use_bias:
            out = out + self.bias
        return out


class ConvDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, size, kernel_size=3, padding=1):
        super().__init__()
        assert size % 2 == 0
        self.conv1 = Conv2dBias(in_ch, in_ch, kernel_size=kernel_size, size=size, stride=1, padding=padding, use_bias=True)
        self.conv2 = Conv2dBias(in_ch, out_ch, kernel_size=kernel_size, size=size//2, stride=2, padding=padding, use_bias=True)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv_skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, padding=0, bias=False)

    def forward(self, x):
        x_skip = self.conv_skip(x)
        x_ = self.conv1(x)
        x_ = self.lrelu(x_)
        x_ = self.conv2(x_)
        x_ = self.lrelu(x_)
        out = x_ + x_skip
        return out


class ConvUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, size, kernel_size=3, padding=1):
        super().__init__()
        assert size % 2 == 0
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = Conv2dBias(in_ch, out_ch, kernel_size=kernel_size, size=size, stride=1, padding=padding, use_bias=True)
        self.conv2 = Conv2dBias(out_ch, out_ch, kernel_size=kernel_size, size=size, stride=1, padding=padding, use_bias=True)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv_skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.upsample(x)
        x_skip = self.conv_skip(x)
        x_ = self.conv1(x)
        x_ = self.lrelu(x_)
        x_ = self.conv2(x_)
        x_ = self.lrelu(x_)
        out = x_ + x_skip
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, size, kernel_size=3, padding=1):
        super().__init__()
        assert size % 2 == 0
        self.conv1 = Conv2dBias(in_ch, in_ch, kernel_size=kernel_size, size=size, stride=1, padding=padding, use_bias=True)
        self.conv2 = Conv2dBias(in_ch, out_ch, kernel_size=kernel_size, size=size, stride=1, padding=padding, use_bias=True)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv_skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_skip = self.conv_skip(x)
        x_ = self.conv1(x)
        x_ = self.lrelu(x_)
        x_ = self.conv2(x_)
        x_ = self.lrelu(x_)
        out = x_ + x_skip
        return out


class Conv1dBias(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, size, stride, padding, use_bias=True, *args, **kwargs):
        super().__init__(in_ch, out_ch, bias=False, kernel_size=kernel_size, stride=stride, padding=padding, *args, **kwargs)
        self.use_bias = use_bias
        if self.use_bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(1, out_ch, size), requires_grad=True))

    def forward(self, x):
        out = F.conv1d(x, self.weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) 
        if self.use_bias:
            out = out + self.bias
        return out


class Conv1dDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, size, kernel_size=3, padding=1):
        super().__init__()
        # assert size % 2 == 0
        size_half = size // 2 if size % 2 == 0 else (size + 1) // 2
        self.conv1 = Conv1dBias(in_ch, in_ch, kernel_size=kernel_size, size=size, stride=1, padding=padding, use_bias=True)
        self.conv2 = Conv1dBias(in_ch, out_ch, kernel_size=kernel_size, size=size_half, stride=2, padding=padding, use_bias=True)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv_skip = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=2, padding=0, bias=False)

    def forward(self, x):
        x_skip = self.conv_skip(x)
        x_ = self.conv1(x)
        x_ = self.lrelu(x_)
        x_ = self.conv2(x_)
        x_ = self.lrelu(x_)
        out = x_ + x_skip
        return out


class Conv1dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, size, kernel_size=3, padding=1):
        super().__init__()
        assert size % 2 == 0
        self.conv1 = Conv1dBias(in_ch, in_ch, kernel_size=kernel_size, size=size, stride=1, padding=padding, use_bias=True)
        self.conv2 = Conv1dBias(in_ch, out_ch, kernel_size=kernel_size, size=size, stride=1, padding=padding, use_bias=True)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv_skip = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_skip = self.conv_skip(x)
        x_ = self.conv1(x)
        x_ = self.lrelu(x_)
        x_ = self.conv2(x_)
        x_ = self.lrelu(x_)
        out = x_ + x_skip
        return out
