# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:56:42 2020

@author: Karthik
"""

import torch

from .model_utils import Upsample, Downsample
from torch import nn


class UNet(nn.Module):

    def __init__(self, num_classes, criterion):
        super(UNet, self).__init__()
        #criterion
        self.criterion = criterion
        # downsampling images
        self.down1 = Downsample(in_channels = 3,
                                out_channels = 64,
                                kernel_size = 3,
                                padding = 1)

        self.down2 = Downsample(in_channels = 64,
                                out_channels = 128,
                                kernel_size = 3,
                                padding = 1)

        self.down3 = Downsample(in_channels = 128,
                                out_channels = 256,
                                kernel_size = 3,
                                padding = 1)

        self.down4 = Downsample(in_channels = 256,
                                out_channels = 512,
                                kernel_size = 3,
                                padding = 1)

        self.down5 = Downsample(in_channels = 512,
                                out_channels = 1204,
                                kernel_size = 3,
                                padding = 1)

        
        self.up4 = Upsample(in_channels = 1024,
                            skip_channels = 512,
                            out_channels = 512,
                            kernel_size = 2,
                            stride = 2,
                            padding = 0)

        self.up3 = Upsample(in_channels = 512,
                            skip_channels = 256,
                            out_channels = 256,
                            kernel_size = 2,
                            stride = 2,
                            padding = 0)

        self.up2 = Upsample(in_channels = 256,
                            skip_channels = 128,
                            out_channels = 128,
                            kernel_size = 2,
                            stride = 2,
                            padding = 0)

        self.up1 = Upsample(in_channels = 128,
                            skip_channels = 64,
                            out_channels = 64,
                            kernel_size = 2,
                            stride = 2,
                            padding = 0)

        self.conv = nn.Conv2d(in_channels = 64,
                              out_channels = num_classes,
                              kernel_size = 3,
                              padding = 1)

    def forward(self, inp, gt = None):
        # downsample
        down1 = self.down1(inp)
        print("-----------------------------> {}".format(down1.shape))
        down2 = self.down2(down1)
        print("-----------------------------> {}".format(down2.shape))
        down3 = self.down3(down2)
        print("-----------------------------> {}".format(down3.shape))
        down4 = self.down4(down3)
        print("-----------------------------> {}".format(down4.shape))
        down5 = self.down5(down4)
        print("-----------------------------> {}".format(down5.shape))
        # upsample
        up4 = self.up4(down5, down4)
        up3 = self.up3(up4, down3)
        up2 = self.up2(up3, down2)
        up1 = self.up1(up2, down1)

        gt_pred = self.conv(up1)

        if gt is not None:
            return self.criterion(inp, gt)

        return gt_pred