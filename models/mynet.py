# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:28:32 2020

@author: Karthik
"""
import torch
import torch.nn.functional as F

from torch import nn
from torchvision import models
from model_utils import Upsample, Downsample

#Basic decoder block
class Dconv(nn.Module):

    def __init__(self, planes, outplanes, kernel_size, stride=1, padding = 0,
                 norm_layer=None):
        super(Dconv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.dconv1 = nn.ConvTranspose2d(planes, outplanes, kernel_size,
                                         stride, padding)
        self.bn1 = norm_layer(outplanes)
        self.stride = stride

    def forward(self, x):
        out = self.dconv1(x)
        out = self.bn1(out)

        return out

class SegNet(nn.Module):
    
    def __init__(self, num_classes, criterion):
        super(SegNet, self).__init__()
        self.criterion = criterion
        resnet18 = models.resnet18(pretrained = True)
        #Encoder consists of pretrained resnet architecture without the
        #classification layer
        self.encoder = nn.Sequential(*list(resnet18.children())[:-2])
        #Decoder consists of multiple layers of transposed convolution and batch
        #normalization layers
        self.decoder = nn.Sequential(
            Dconv(512, 256, 2, stride= 2),
            Dconv(256, 128, 2, stride = 2),
            Dconv(128, 128, 2, stride = 2),
            nn.Conv2d(128, 1, 3)
        )

    def forward(self, input, gt = None):
        x = self.encoder(input)
        x = self.decoder(x)
        #The output tensor from the decoder is resized to input image size
        #to make it easier for loss calculation using binary cross entropy loss
        x = F.interpolate(x, input.size()[2:], mode = "bilinear")
        if gt is not None:
            return self.criterion(x, gt)
        return x