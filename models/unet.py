# -*- coding: utf-8 -*-
import torch
from torch import nn
try:
    from model_utils import conv2d, deconv2d
except:
    from .model_utils import conv2d, deconv2d

class UNet(nn.Module):

    def __init__(self, num_classes):
        super(UNet, self).__init__()

        self.conv1 = conv2d(3, 64 , kernel_size=3)
        self.conv1_1 = conv2d(64 , 64 , kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = conv2d(64, 128, kernel_size=3)
        self.conv2_1 = conv2d(128, 128, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = conv2d(128, 256, kernel_size=3)
        self.conv3_1 = conv2d(256, 256, kernel_size=3)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = conv2d(256, 512, kernel_size=3)
        self.conv4_1 = conv2d(512, 512, kernel_size=3)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = conv2d(512, 1024, kernel_size=3)
        self.conv5_1 = conv2d(1024, 1024, kernel_size=3)

        self.deconv4 = deconv2d(1024, 512)
        self.conv6 = conv2d(1024, 512, kernel_size=3)
        self.conv6_1 = conv2d(512, 512, kernel_size=3)

        self.deconv3 = deconv2d(512, 256)
        self.conv7 = conv2d(512, 256, kernel_size=3)
        self.conv7_1 = conv2d(256, 256, kernel_size=3) 

        self.deconv2 = deconv2d(256, 128)
        self.conv8 = conv2d(256, 128, kernel_size=3)
        self.conv8_1 = conv2d(128, 128, kernel_size=3)

        self.deconv1 = deconv2d(128, 64)
        self.conv9 = conv2d(128, 64, kernel_size=3)
        self.conv9_1 = conv2d(64, 64, kernel_size=3)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self.softmax = nn.Softmax2d()

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def network_output(self, input_image):
        out_conv1 = self.conv1_1(self.conv1(input_image))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))

        out_deconv4 = self.deconv4(out_conv5)
        concat4 = torch.cat((out_conv4, out_deconv4), 1)
        out_conv6 = self.conv6_1(self.conv6(concat4))

        out_deconv3 = self.deconv3(out_conv6)
        concat3 = torch.cat((out_conv3, out_deconv3), 1)
        out_conv7 = self.conv7_1(self.conv7(concat3))

        out_deconv2 = self.deconv2(out_conv7)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_conv8 = self.conv8_1(self.conv8(concat2))

        out_deconv1 = self.deconv1(out_conv8)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_conv9 = self.conv9_1(self.conv9(concat1))

        final_out = self.softmax(self.final_conv(out_conv9))

        return final_out

    def forward(self, input_image, target):
        net_out = self.network_output(input_image)
        loss = self.cross_entropy_loss(net_out, target)
        return loss