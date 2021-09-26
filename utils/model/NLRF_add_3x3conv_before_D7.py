import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .NL import NONLocalBlock2D

class NLRF_block(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(NLRF_block, self).__init__()

        self.non_local_receptive_field_operation = NONLocalBlock2D(in_channels = 512)

        self.inplanes = inplanes
        self.outplanes = outplanes

        rates = [1, 3, 7]

        self.dilated_convolution_0 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                                            stride=1, padding=0)
        self.dilated_convolution_1 = nn.Conv2d(inplanes, outplanes, kernel_size=3,
                                            stride=1, padding=rates[0], dilation=rates[0])
        self.dilated_convolution_2 = nn.Conv2d(inplanes, outplanes, kernel_size=3,
                                            stride=1, padding=rates[1]-1, dilation=rates[1]-1)
        self.dilated_convolution_3 = nn.Conv2d(inplanes, outplanes, kernel_size=3,
                                            stride=1, padding=rates[2], dilation=rates[2])

    def forward(self, x):
        x1 = self.dilated_convolution_0(x)
        x1 = self.dilated_convolution_1(x1)

        x2 = self.dilated_convolution_1(x)
        x2 = self.dilated_convolution_2(x2)

        x3 = self.dilated_convolution_1(x)
        x3 = self.dilated_convolution_3(x3)

        x13 = torch.cat((x1,x3),1)
        x13 = self.non_local_receptive_field_operation(x13)

        x1 = x1 + x13
        x = x1 + x2 + x3

        return x