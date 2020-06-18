# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:22:02 2020

@author: Karthik
"""
import torch
from torch import nn
        

class Conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 bias = True,
                 dropout = False
                ):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size = kernel_size,
                                   stride = stride,
                                   padding = padding,
                                   bias = bias)
        self.bn =  nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, inp):
        x = self.conv(inp)
        x = self.bn(x)
        x = self.relu(x)
        if self.drop:
            x = self.dropout(x)
        return x


class Dconv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 1,
                 padding = 0,
                 output_padding = 0,
                 bias = False,
                 dropout = False
                ):
        super(Dconv2d, self).__init__()
        self.dconv = nn.ConvTranspose2d(in_channels,
                                   out_channels,
                                   kernel_size = kernel_size,
                                   stride = stride,
                                   padding = padding,
                                   output_padding = output_padding,
                                   bias = bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace = True)
        self.drop = dropout

    def forward(self, inp):
        x = self.dconv(inp)
        x = self.bn(x)
        x = self.relu(x)
        if self.drop :
            x = self.dropout(x)
        return x
    

class Downsample(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 bias = False,
                 dropout = False,
                 size = 2
               ):
        super(Downsample, self).__init__()
        self.conv1 = Conv2d(in_channels = in_channels,
                            out_channels = out_channels,
                            kernel_size = kernel_size,
                            stride = stride,
                            padding = padding,
                            bias = bias,
                            dropout = dropout)
        self.conv2 = Conv2d(in_channels = out_channels,
                            out_channels = out_channels,
                            kernel_size = kernel_size,
                            stride = stride,
                            padding = padding,
                            bias = bias,
                            dropout = dropout)
        self.maxpool =  nn.MaxPool2d(size)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.conv2(x)
        return self.maxpool(x)

class Upsample(nn.Module):
    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 output_padding = 0,
                 bias = True,
                 dropout = False,
              ):
        super(Upsample, self).__init__()
        self.dconv = Dconv2d(in_channels = in_channels,
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = padding,
                                output_padding = output_padding,
                                bias = bias,
                                dropout = dropout)
        #Concat before sending it to conv layer
        self.conv = Conv2d(in_channels + skip_channels,
                           out_channels = out_channels,
                           kernel_size = kernel_size,
                           stride = stride,
                           padding = padding,
                           bias = bias)

    def forward(self, x, x_con):
        x = self.dconv(x)
        x = torch.cat(tuple(x, x_con), dim = 1)
        x = self.conv(x)
        return x