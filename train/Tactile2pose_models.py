import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import pandas as pd
from Tactile2pose_dataLoader import *
from torch.nn.parameter import Parameter
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import argparse

class SpatialSoftmax3D(torch.nn.Module):
    def __init__(self, height, width, depth, channel, lim=[0., 1., 0., 1., 0., 1.], temperature=None, data_format='NCHWD'):
        super(SpatialSoftmax3D, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.depth = depth
        self.channel = channel
        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.
        pos_y, pos_x, pos_z = np.meshgrid(
            np.linspace(lim[0], lim[1], self.width),
            np.linspace(lim[2], lim[3], self.height),
            np.linspace(lim[4], lim[5], self.depth))
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width * self.depth)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width * self.depth)).float()
        pos_z = torch.from_numpy(pos_z.reshape(self.height * self.width * self.depth)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
        self.register_buffer('pos_z', pos_z)
    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWDC':
            feature = feature.transpose(1, 4).tranpose(2, 4).tranpose(3,4).reshape(-1, self.height * self.width * self.depth)
        else:
            feature = feature.reshape(-1, self.height * self.width * self.depth).to('cpu')
        softmax_attention = feature.detach()
        # softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        heatmap = softmax_attention.reshape(-1, self.channel, self.height, self.width, self.depth).to('cpu')

        eps = 1e-6
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)/(torch.sum(softmax_attention, dim=1, keepdim=True) + eps)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)/(torch.sum(softmax_attention, dim=1, keepdim=True) + eps)
        expected_z = torch.sum(self.pos_z * softmax_attention, dim=1, keepdim=True)/(torch.sum(softmax_attention, dim=1, keepdim=True) + eps)
        expected_xyz = torch.cat([expected_x, expected_y, expected_z], 1)
        feature_keypoints = expected_xyz.reshape(-1, self.channel, 3)
        return feature_keypoints, heatmap

class CNN_intelligent_carpet(nn.Module):
    def __init__(self, windowSize):
        super(CNN_intelligent_carpet, self).__init__()  # tactile 96*96
        if windowSize == 0:
            self.conv_0 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32))
        else:
            self.conv_0 = nn.Sequential(
                nn.Conv2d(windowSize, 32, kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32))

        self.conv_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2))  # 32 * 32

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),)

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2))  # 16 * 16

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),)

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2))  # 8 * 8

        self.convTrans_0 = nn.Sequential(
            nn.Conv3d(1025, 1025, kernel_size=(3, 3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(1025))

        self.convTrans_1 = nn.Sequential(
            nn.Conv3d(1025, 512, kernel_size=(3, 3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(512))

        self.convTrans_00 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=(2, 2, 2), stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm3d(256))

        self.convTrans_2 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=(3, 3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(128))

        self.convTrans_3 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(64))

        self.convTrans_4 = nn.Sequential(
            nn.Conv3d(64, 12, kernel_size=(3, 3, 3), padding=1),
            nn.Sigmoid())

    def forward(self, input):
        output = self.conv_0(input)
        output = self.conv_1(output)
        output = self.conv_2(output)
        output = self.conv_3(output)
        output = self.conv_4(output)
        output = self.conv_5(output)

        output = output.reshape(output.shape[0], output.shape[1], output.shape[2], output.shape[3], 1)
        output = output.repeat(1, 1, 1, 1, 8)

        layer = torch.zeros(output.shape[0], 1, output.shape[2], output.shape[3], output.shape[4]).cuda(0)
        for i in range(layer.shape[4]):
            layer[:, :, :, :, i] = i
        layer = layer / (layer.shape[4] - 1)
        output = torch.cat((output, layer), axis=1)

        output = self.convTrans_0(output)
        output = self.convTrans_1(output)
        output = self.convTrans_00(output)
        output = self.convTrans_2(output)
        output = self.convTrans_3(output)
        output = self.convTrans_4(output)

        return output

class LSTMCNN_intelligence(nn.Module):
    def __init__(self, window_size):
        super(LSTMCNN_intelligence, self).__init__()
        # self.cnn = CNN()
        self.cnn = CNN_intelligent_carpet(window_size)
        self.rnn = nn.LSTM(
            input_size=256, #256
            hidden_size=64,
            num_layers=1,
            batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(64, 12),
        )

    def forward(self, input):
        # input = input.unsqueeze(2)
        # batch_size, timestamps, C, H, W = input.size()
        # cnn_in = input.view(batch_size * timestamps, C, H, W)
        # cnn_out = self.cnn(cnn_in)
        batch_size, timestamps, H, W = input.size()
        # cnn_in = input.view(batch_size * timestamps, H, W)
        cnn_out = self.cnn(input)
        rnn_in = cnn_out.view(batch_size, timestamps, -1)
        rnn_out, (h_n, h_c) = self.rnn(rnn_in)
        rnn_out2 = self.linear(rnn_out[:,-1,:])

        return rnn_out2

class CNN_intelligent_carpet_short(nn.Module):
    def __init__(self, windowSize):
        super(CNN_intelligent_carpet_short, self).__init__()  # tactile 96*96
        if windowSize == 0:
            self.conv_0 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32))
        else:
            self.conv_0 = nn.Sequential(
                nn.Conv2d(windowSize, 32, kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=2)) # 32 * 32

        self.conv_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2))  # 16 * 16

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2)) # 8 * 8

        # self.conv_3 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm2d(256),
        #     nn.MaxPool2d(kernel_size=2))  # 8 * 8


        self.fc1 = nn.Linear(8 * 8 * 64, windowSize * 16, bias=True)

    def forward(self, input):
        output = self.conv_0(input)
        output = self.conv_1(output)
        output = self.conv_2(output)
        # output = self.conv_3(output)
        output = output.view(-1, 8 * 8 * 64)
        output = self.fc1(output)

        return output

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(10, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 8 * 32, 64)
        # self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 8 * 8 * 32)
        x = torch.relu(self.fc1(x))
        # x = self.fc2(x)
        return x

class LSTMCNN(nn.Module):
    def __init__(self, window_size):
        super(LSTMCNN, self).__init__()
        # self.cnn = CNN()
        self.cnn = CNN_intelligent_carpet_short(window_size)
        self.rnn = nn.LSTM(
            input_size=16, #256
            hidden_size=16,
            num_layers=1,
            batch_first=True)
        self.linear = nn.Sequential(
            # nn.Linear(16, 1), # for single muscle
            nn.Linear(16, 12) # for all muscle
        )

    def forward(self, input):
        # input = input.unsqueeze(2)
        # batch_size, timestamps, C, H, W = input.size()
        # cnn_in = input.view(batch_size * timestamps, C, H, W)
        # cnn_out = self.cnn(cnn_in)
        batch_size, timestamps, H, W = input.size()
        # cnn_in = input.view(batch_size * timestamps, H, W)
        cnn_out = self.cnn(input)
        rnn_in = cnn_out.view(batch_size, timestamps, -1)
        rnn_out, (h_n, h_c) = self.rnn(rnn_in)
        rnn_out2 = self.linear(rnn_out[:,-1,:])

        return rnn_out2