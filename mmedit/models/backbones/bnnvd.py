# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init
from mmcv.runner import _load_checkpoint, load_state_dict

from mmedit.models.common import make_layer
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger

from .DABC import BinaryConv2dSkip1x1, BNNDownSample, BNNUpSample, BNNSkipUpSample
from .DABC import DABCConv2d as BinaryConv2d

import pdb


import io
import logging
import os
import os.path as osp
import pkgutil
import re
import time
import warnings
from collections import OrderedDict
from importlib import import_module
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, Optional, Tuple, Union

def load_checkpoint(
    model: torch.nn.Module,
    filename: str,
    map_location: Union[str, Callable, None] = None,
    strict: bool = False,
    logger: Optional[logging.Logger] = None,
    revise_keys: list = [(r'^module\.', '')]) -> Union[dict, OrderedDict]:
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Default: strip
            the prefix 'module.' by [(r'^module\\.', '')].

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')

    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
            for k, v in state_dict.items()})
        
    state_dict.pop('step_counter')

    # Keep metadata in state_dict
    state_dict._metadata = metadata

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class DABCWithInputConv(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                BinaryConv2d, num_blocks,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                bias=False))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)
    
class BBCUWithBinarizedInputConv(nn.Module):
    def __init__(self, in_channels, in_groups, out_channels=64, num_blocks=30, kernel_size=3):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(BinaryConv2dSkip1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            groups=in_groups,
            bias=False
        ))

        # residual blocks
        main.append(
            make_layer(
                BinaryConv2d, num_blocks,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=False))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)
    

class BNNUnet(nn.Module):
    def __init__(self, n_feat=[24, 36, 48], n_block=[1, 3, 3]):
        super(BNNUnet, self).__init__()
        # Encoder
        self.encoder_level1 = make_layer(
                BinaryConv2d, n_block[0],
                in_channels=n_feat[0],
                out_channels=n_feat[0],
                kernel_size=3,
                bias=False)
        self.down12 = BNNDownSample(n_feat[0], n_feat[1])
        
        self.encoder_level2 = make_layer(
                BinaryConv2d, n_block[1],
                in_channels=n_feat[1],
                out_channels=n_feat[1],
                kernel_size=3,
                bias=False)
        self.down23 = BNNDownSample(n_feat[1], n_feat[2])
        
        self.encoder_level3 = make_layer(
                BinaryConv2d, n_block[2],
                in_channels=n_feat[2],
                out_channels=n_feat[2],
                kernel_size=3,
                bias=False)
        
        # Decoder
        self.decoder_level3 = make_layer(
                BinaryConv2d, n_block[2],
                in_channels=n_feat[2],
                out_channels=n_feat[2],
                kernel_size=3,
                bias=False)
        
        self.skip_conv2 = BinaryConv2d(n_feat[1], n_feat[1], 3)
        self.up32 = BNNSkipUpSample(n_feat[2], n_feat[1])
        self.decoder_level2 = make_layer(
                BinaryConv2d, n_block[1],
                in_channels=n_feat[1],
                out_channels=n_feat[1],
                kernel_size=3,
                bias=False)
        
        self.skip_conv1 = BinaryConv2d(n_feat[0], n_feat[0], 3)
        self.up21 = BNNSkipUpSample(n_feat[1], n_feat[0])
        self.decoder_level1 = make_layer(
                BinaryConv2d, n_block[0],
                in_channels=n_feat[0],
                out_channels=n_feat[0],
                kernel_size=3,
                bias=False)
        
    def forward(self, x):
        shortcut = x 
        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)

        dec3 = self.decoder_level3(enc3)
        x = self.up32(dec3, self.skip_conv2(enc2))
        dec2 = self.decoder_level2(x)
        x = self.up21(dec2, self.skip_conv1(enc1))
        dec1 = self.decoder_level1(x)
        return dec1 + shortcut
    
class ShiftBlock1(nn.Module):
    """
    Channel 维度进行 shift
    """
    def __init__(self, n_feat, kernel_size, num_frame=3):
        super(ShiftBlock1, self).__init__()
        self.T = num_frame
        self.body = BinaryConv2d(n_feat, n_feat, kernel_size)

    def channel_shift(self, x, div=2, reverse=False):
        B, T, C, H, W = x.shape
        slice_c = C // div
        if reverse:
            slice_c = -slice_c
        
        y = x.view(B, T*C, H, W)
        y = torch.roll(y, slice_c, 1).view(B, T, C, H, W)

        return y

    def forward(self, x, reverse=False):
        _, C, H, W = x.shape
        x = self.channel_shift(x.view(-1, self.T, C, H, W), reverse=reverse)
        res = self.body(x.view(-1, C, H, W))
        return res
    
class ShiftBlock2(nn.Module):
    """
    Channel, Spatial 维度进行 shift
    """
    def __init__(self, n_feat, kernel_size=5, num_frame=3):
        super(ShiftBlock2, self).__init__()
        number = n_feat // 2 // 8
        self.number = number
        self.T = num_frame
        self.encoder_1 = BBCUWithBinarizedInputConv(n_feat + 8*self.number, 1, n_feat, 1, kernel_size)
        self.encoder_2 = BBCUWithBinarizedInputConv(n_feat + 8*self.number, 1, n_feat, 1, kernel_size)
        self.encoder_3 = BBCUWithBinarizedInputConv(n_feat + 8*self.number, 1, n_feat, 1, kernel_size)
        self.encoder_4 = BBCUWithBinarizedInputConv(n_feat + 8*self.number, 1, n_feat, 1, kernel_size)

    def spatial_shift(self, hw):
        n2 = (self.number-1)//2
        n1 = self.number-2*n2
        s = 4
        s_list = []
        _, _, _, H, W = hw.shape
        s_out = torch.zeros_like(hw)
        s_out[:,0*n2:1*n2,s*2:,s*2:] = hw[:,0*n2:1*n2,:-s*2,:-s*2]
        s_out[:,1*n2:2*n2,s*2:,s:] = hw[:,1*n2:2*n2,:-s*2,:-s]
        s_out[:,2*n2:3*n2,s*2:,0:] = hw[:,2*n2:3*n2,:-s*2,:]
        s_out[:,3*n2:4*n2,s*2:,0:-s] = hw[:,3*n2:4*n2,:-s*2,s:]
        s_out[:,4*n2:5*n2,s*2:,0:-s*2] = hw[:,4*n2:5*n2,:-s*2,s*2:]

        s_out[:,5*n2:6*n2,0:-s*2,s*2:] = hw[:,5*n2:6*n2,s*2:,:-s*2]
        s_out[:,6*n2:7*n2,0:-s*2,s:] = hw[:,6*n2:7*n2,s*2:,:-s]
        s_out[:,7*n2:8*n2,0:-s*2,0:] = hw[:,7*n2:8*n2,s*2:,:]
        s_out[:,8*n2:9*n2,0:-s*2,0:-s] = hw[:,8*n2:9*n2,s*2:,s:]
        s_out[:,9*n2:10*n2,0:-s*2,0:-s*2] = hw[:,9*n2:10*n2,s*2:,s*2:]

        s_out[:,10*n2:11*n2,s:,s*2:] = hw[:,10*n2:11*n2,  :-s,:-s*2]
        s_out[:,11*n2:12*n2,s:,0:-s*2] = hw[:,11*n2:12*n2,:-s,s*2:]
        s_out[:,12*n2:13*n2,:,s*2:] = hw[:,12*n2:13*n2,  :,:-s*2]
        s_out[:,13*n2:14*n2,:,0:-s*2] = hw[:,13*n2:14*n2,:,s*2:]
        s_out[:,14*n2:15*n2,0:-s,s*2:] = hw[:,14*n2:15*n2,  s:,:-s*2]
        s_out[:,15*n2:16*n2,0:-s,0:-s*2] = hw[:,15*n2:16*n2,s:,s*2:]
        s_out[:,16*n2+0*n1:16*n2+1*n1,s:,s:] = hw[:,16*n2+0*n1:16*n2+1*n1,:-s,:-s]
        s_out[:,16*n2+1*n1:16*n2+2*n1,s:,0:] = hw[:,16*n2+1*n1:16*n2+2*n1,:-s,:]
        s_out[:,16*n2+2*n1:16*n2+3*n1,s:,0:-s] = hw[:,16*n2+2*n1:16*n2+3*n1,:-s,s:]
        s_out[:,16*n2+3*n1:16*n2+4*n1,:,s:] = hw[:,16*n2+3*n1:16*n2+4*n1,:,:-s]
        s_out[:,16*n2+4*n1:16*n2+5*n1,:,0:-s] = hw[:,16*n2+4*n1:16*n2+5*n1,:,s:]
        s_out[:,16*n2+5*n1:16*n2+6*n1,0:-s,s:] = hw[:,16*n2+5*n1:16*n2+6*n1,s:,:-s]
        s_out[:,16*n2+6*n1:16*n2+7*n1,0:-s,0:] = hw[:,16*n2+6*n1:16*n2+7*n1,s:,:]
        s_out[:,16*n2+7*n1:16*n2+8*n1,0:-s,0:-s] = hw[:,16*n2+7*n1:16*n2+8*n1,s:,s:]
        return s_out 

    def channel_spatial_shift(self, x, div=2, reverse=False):
        B, T, C, H, W = x.shape
        slice_c = C // div
        if reverse:
            slice_c = -slice_c
        
        y = x.view(B, T*C, H, W)
        y = torch.roll(y, slice_c, 1).view(B, T, C, H, W)

        if reverse == False:
            hw = y[:, :, 0:8*self.number, :, :]
        else:
            hw = y[:, :, -8*self.number:, :, :]
        hw = self.spatial_shift(hw)

        return torch.cat((y, hw), dim=2)

    def forward(self, x):
        _, C, H, W = x.shape
        x = self.channel_spatial_shift(x.view(-1, self.T, C, H, W), reverse=False)
        C1 = x.shape[2]
        x = self.encoder_1(x.view(-1, C1, H, W))
        x = self.channel_spatial_shift(x.view(-1, self.T, C, H, W), reverse=True)
        x = self.encoder_2(x.view(-1, C1, H, W))
        x = self.channel_spatial_shift(x.view(-1, self.T, C, H, W), reverse=False)
        x = self.encoder_3(x.view(-1, C1, H, W))
        x = self.channel_spatial_shift(x.view(-1, self.T, C, H, W), reverse=True)
        x = self.encoder_4(x.view(-1, C1, H, W))

        return x


class ShiftEncoder(nn.Module):
    def __init__(self, n_feat=[24, 80, 80, 80]):
        super(ShiftEncoder, self).__init__()
        self.conv_in = BinaryConv2d(n_feat[0], n_feat[0], kernel_size=3)
        self.encoder_level0 = ShiftBlock1(n_feat[0], kernel_size=3)
        self.encoder_level0_1 = ShiftBlock1(n_feat[0], kernel_size=3)
        self.down01 = BNNDownSample(n_feat[0], n_feat[1])

        self.encoder_level1 = ShiftBlock1(n_feat[1], kernel_size=3)
        self.encoder_level1_1 = ShiftBlock1(n_feat[1], kernel_size=3)
        self.down12 = BNNDownSample(n_feat[1], n_feat[2])

        self.encoder_level2 = BinaryConv2d(n_feat[2], n_feat[2], kernel_size=3)
        self.encoder_level2_1 = BinaryConv2d(n_feat[2], n_feat[2], kernel_size=3)
        self.down23 = BNNDownSample(n_feat[2], n_feat[3])

        self.encoder_level3 = BinaryConv2d(n_feat[3], n_feat[3], kernel_size=3)
        self.encoder_level3_1 = BinaryConv2d(n_feat[3], n_feat[3], kernel_size=3)
        
        self.skip_conv0 = BinaryConv2d(n_feat[0], n_feat[0], kernel_size=3)
        self.skip_conv1 = BinaryConv2d(n_feat[1], n_feat[1], kernel_size=3)
        self.skip_conv2 = BinaryConv2d(n_feat[2], n_feat[2], kernel_size=3)

        self.decoder_level3 = ShiftBlock2(n_feat[3], kernel_size=5)
        self.decoder_level3_1 = ShiftBlock2(n_feat[3], kernel_size=5)
        self.up32 = BNNSkipUpSample(n_feat[3], n_feat[2])

        self.decoder_level2 = ShiftBlock2(n_feat[2], kernel_size=5)
        self.decoder_level2_1 = ShiftBlock2(n_feat[2], kernel_size=5)
        self.up21 = BNNSkipUpSample(n_feat[2], n_feat[1])

        self.decoder_level1 = ShiftBlock2(n_feat[1], kernel_size=5)
        self.decoder_level1_1 = ShiftBlock2(n_feat[1], kernel_size=5)
        self.decoder_level1_2 = ShiftBlock2(n_feat[1], kernel_size=5)
        self.up10 = BNNUpSample(n_feat[1], n_feat[0])

        self.conv_hr = BinaryConv2dSkip1x1(2*n_feat[0], n_feat[0], 3)
        self.conv_out = BinaryConv2d(n_feat[0], n_feat[0], 3)

    def forward(self, x):
        n, t, c, h, w = x.shape
        x = self.conv_in(x.view(n*t, c, h, w))
        enc0 = self.encoder_level0(x, reverse=False)
        enc0 = self.encoder_level0_1(enc0, reverse=True)
        enc0_down = self.down01(enc0)

        enc1 = self.encoder_level1(enc0_down, reverse=False)
        enc1 = self.encoder_level1_1(enc1, reverse=True)
        enc1_down = self.down12(enc1)

        enc2 = self.encoder_level2(enc1_down)
        enc2 = self.encoder_level2_1(enc2)
        enc2_down = self.down23(enc2)

        enc3 = self.encoder_level3(enc2_down)
        enc3 = self.encoder_level3_1(enc3)

        dec3 = self.decoder_level3(enc3)
        dec3 = self.decoder_level3_1(dec3)
        dec3_up = self.up32(dec3, self.skip_conv2(enc2))

        dec2 = self.decoder_level2(dec3_up)
        dec2 = self.decoder_level2_1(dec2)
        dec2_up = self.up21(dec2, self.skip_conv1(enc1))

        dec1 = self.decoder_level1(dec2_up)
        dec1 = self.decoder_level1_1(dec1)
        dec1 = self.decoder_level1_2(dec1)
        out = self.conv_out(self.conv_hr(torch.cat((self.up10(dec1), self.skip_conv0(x)), dim=1))).view(n, t, -1, h, w)

        return out[:, 0, :, :, :], out[:, 1, :, :, :], out[:, 2, :, :, :]


@BACKBONES.register_module()
class BNNVD(nn.Module):
    """ Binarized Video Denoising
    ShiftBlock2 有4个encoder
    """

    def __init__(self,
                 in_channels=4,
                 mid_channels=24,
                 feat_extract_blocks=3,
                 num_unets=1,
                 unet_n_feat=[24, 48, 96],
                 unet_n_block=[1, 3, 3],
                 stage1_n_feat=[24, 48, 48, 48],
                 task='Raw2Raw'):

        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.task = task

        # 按照BBCU的方法初始化一个k，固定值
        self.k = self.k = (130*mid_channels) / 64.0
        self.feat_extract = DABCWithInputConv(in_channels, mid_channels, feat_extract_blocks)

        self.stage0 = make_layer(
            BNNUnet, num_unets,
            n_feat=unet_n_feat,
            n_block=unet_n_block
        )

        self.stage1 = ShiftEncoder(stage1_n_feat)

        self.fusion = BinaryConv2dSkip1x1(3*mid_channels, mid_channels, 3)
        self.stage2 = make_layer(
            BNNUnet, num_unets,
            n_feat=unet_n_feat,
            n_block=unet_n_block
        )

        self.conv_out = nn.Conv2d(mid_channels, 4, 3, 1, 1, bias=True)

    def forward_test(self, lqs):
        """
        Without feature cache
        """

        n, t, c, h, w = lqs.size()
        lqs = lqs * self.k

        feat_in_l = self.feat_extract(lqs[:, 0, :, :, :])
        feat_l = self.stage0(feat_in_l)
        feat_in_m = self.feat_extract(lqs[:, 1, :, :, :])
        feat_m = self.stage0(feat_in_m)

        outputs = []

        for i in range(2, t):
            feat_in_r = self.feat_extract(lqs[:, i, :, :, :])
            feat_r = self.stage0(feat_in_r)
            feat_in = feat_in_l.clone()
            feat_stage0 = feat_l.clone()
            feat_stage1, feat_l, feat_m = self.stage1(torch.stack([feat_l, feat_m, feat_r], dim=1))
            feat_in_l = feat_in_m
            feat_in_m = feat_in_r
            fusion_in = torch.cat([feat_in, feat_stage0, feat_stage1], dim=1)
            out = lqs + self.conv_out(self.stage2(self.fusion(fusion_in)))
            outputs.append(out)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs / self.k


    def forward(self, lqs):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        n, t, c, h, w = lqs.size()
        lqs = lqs * self.k

        # stage0 feature extraction
        feat_in = self.feat_extract(lqs.view(-1, c, h, w))
        stage0_feat = self.stage0(feat_in)
        stage0_feat = stage0_feat.view(n, t, -1, h, w)

        # stage1 shift enoder
        stage1_feat = []
        feat_l = stage0_feat[:, 0, :, :, :]
        feat_m = stage0_feat[:, 1, :, :, :]

        for i in range(2, t):
            feat_r = stage0_feat[:, i, :, :, :]
            out, feat_l, feat_m = self.stage1(torch.stack([feat_l, feat_m, feat_r], dim=1))
            stage1_feat.append(out)

        stage1_feat.append(feat_l)
        stage1_feat.append(feat_m)
        stage1_feat = torch.stack(stage1_feat, dim=1)

        # stage2
        fusion_in = torch.cat([feat_in, stage0_feat.view(n*t, -1, h, w), stage1_feat.view(n*t, -1, h, w)], dim=1)
        out = lqs[:, :, :4, :, :] + self.conv_out(self.stage2(self.fusion(fusion_in))).view(n, t, -1, h, w)

        return out / self.k

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether strictly load the pretrained
                model. Default: True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger, revise_keys=[(r'^generator\.', '')])
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
