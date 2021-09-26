import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import cv2
from scipy import ndimage, misc
from skimage.measure import label, regionprops_table
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
from mmcv.cnn import constant_init, kaiming_init

class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DFA_block(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(DFA_block, self).__init__()

        self.inplanes = inplanes
        self.outplanes = outplanes

        self.BasicConv = nn.Sequential(
            nn.Conv2d(in_channels=self.inplanes, out_channels=self.outplanes,
                    kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU(inplace=True)
        )

        self.channel_attention = SELayer(self.outplanes)

    def forward(self, x):

        x_depth = x.data.cpu().numpy()
        x1 = x_depth.copy()
        x2 = x_depth.copy()

        ###########################################################
        entire_depth_feature_original = x1
        entire_depth_feature = x2
        outputs_mask = entire_depth_feature[:, :, 0:64, 0:64]
        outputs_mask[outputs_mask > float(-1)] = 1
        outputs_mask[outputs_mask == float(-1)] = 0

        batch = int(entire_depth_feature.shape[0])
        x_depth_feature_Tips = np.zeros((batch, 64, 64))
        dist_transform = np.zeros((batch, 64, 64))

        for i in range(0, batch):
            temp = entire_depth_feature[i, 0, 0:64, 0:64]
            output_dist_transform = cv2.distanceTransform(temp.astype(np.uint8), cv2.DIST_L2, maskSize=5)
            R = np.max(output_dist_transform)
            dist_transform[i, :] = output_dist_transform

            label_img = label(temp)
            props = regionprops_table(label_img, properties=('centroid',
                                                             'orientation'))
            df = pd.DataFrame(props)
            orientation_value = float(max(df['orientation']))
            angle_in_degrees = orientation_value * (180/np.pi) + 90
            theta = (90+angle_in_degrees)/180*math.pi
            x_c = int(min(df['centroid-0']))
            y_c = int(min(df['centroid-1']))
            x1 = x_c + R * math.cos(theta)
            y1 = y_c - R * math.sin(theta)
            x2 = x_c - R * math.cos(theta)
            y2 = y_c + R * math.sin(theta)
            x1_wrist = x1 + R * abs(math.sin(theta))
            y1_wrist = y1 + R * abs(math.cos(theta))
            x2_wrist = x2 + R * abs(math.sin(theta))
            y2_wrist = y2 + R * abs(math.cos(theta))

            points = [(x1_wrist, y1_wrist), (x2_wrist, y2_wrist)]
            x_coords, y_coords = zip(*points)
            A = np.vstack([x_coords, np.ones(len(x_coords))]).T
            m, c = lstsq(A, y_coords)[0]

            input_position = np.zeros((64, 64))
            result = np.where(input_position == 0)
            input_x = np.array(result[1])
            input_y = np.array(result[0])
            mask = m * input_x + c - input_y
            mask = np.array(mask)
            mask[mask > 0] = 1
            mask[mask <= 0] = 0
            mask = mask.reshape(64, 64)

            if abs(m)<0.75 and np.sum(mask) / (64 * 64) >= 0.45:
                batch_x1 = entire_depth_feature_original[i, 0, 0:64, 0:64]
                x_depth_feature_Tips[i, :] = ((batch_x1 + 1) * mask) - 1
            else:
                batch_x1 = entire_depth_feature_original[i, 0, 0:64, 0:64]
                x_depth_feature_Tips[i, :] = batch_x1

        x_depth_feature_Tips = x_depth_feature_Tips[:, np.newaxis, :]

        ###########################################################

        x_depth_feature1 = torch.from_numpy(x_depth_feature_Tips)
        x_depth_feature1 = x_depth_feature1.type(torch.FloatTensor)
        x_depth_feature1 = x_depth_feature1.cuda()

        ###########################################################

        x_wrist = self.BasicConv(x_depth_feature1)
        x_original = self.BasicConv(x)

        ###########################################################
        x_wrist = self.channel_attention(x_wrist)
        x=x_original+x_wrist
        return x
