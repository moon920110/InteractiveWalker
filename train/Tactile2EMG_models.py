import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import pandas as pd
from Tactile2EMG_dataLoader import *
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import argparse

class CNN_encoder_decoder0906(nn.Module):
    def __init__(self, windowSize): # 0905 ver
        super(CNN_encoder_decoder0906, self).__init__()  # tactile 96*96

        self.conv_0 = nn.Sequential(
            nn.Conv2d(windowSize, 32, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))

        self.conv_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2))  # 16 * 16

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))

        self.conv_3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2))  # 8 * 8


        self.fc1 = nn.Linear(16 * 8 * 8, 128, bias=True)
        self.fc2 = nn.Linear(128, 1, bias = True)

    def forward(self, input):
        output = self.conv_0(input)
        output = self.conv_1(output)
        output = self.conv_2(output)
        output = self.conv_3(output)

        output = output.view(-1, 16 * 8 * 8)
        output = self.fc1(output)
        output = self.fc2(output)

        return output

class CNN_encoder_decoder(nn.Module):
    def __init__(self, windowSize): #0405 ver
        super(CNN_encoder_decoder, self).__init__()  # tactile 96*96

        self.conv_0 = nn.Sequential(
            nn.Conv2d(windowSize, 8, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8))

        self.conv_1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2))  # 16 * 16

        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16))

        self.conv_3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2))  # 8 * 8


        self.fc1 = nn.Linear(8 * 8 * 8, 1, bias=True)

    def forward(self, input):
        output = self.conv_0(input)
        output = self.conv_1(output)
        output = self.conv_2(output)
        output = self.conv_3(output)

        output = output.view(-1, 8 * 8 * 8)
        output = self.fc1(output)

        return output

class CNN_encoder_decoder_classifier(nn.Module):
    def __init__(self, windowSize):
        super(CNN_encoder_decoder_classifier, self).__init__()  # tactile 96*96

        self.conv_0 = nn.Sequential(
            nn.Conv2d(windowSize, 32, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))

        self.conv_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2))  # 16 * 16

        # self.conv_2 = nn.Sequential(
        #     nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm2d(16))

        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2))  # 8 * 8


        self.fc1 = nn.Linear(32 * 8 * 8, 3, bias=True)


    def forward(self, input):
        output = self.conv_0(input)
        output = self.conv_1(output)
        # output = self.conv_2(output)
        output = self.conv_3(output)

        output = output.view(-1, 8 * 8 * 8)
        output = self.fc1(output)

        return output

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
            nn.MaxPool2d(kernel_size=2))  # 48 * 48

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2))  # 24 * 24

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512))

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(5, 5)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024))

        self.conv_6 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2))  # 10 * 10

        self.fc1 = nn.Linear(6 * 6 * 1024, windowSize * 256, bias=True)

    def forward(self, input):
        output = self.conv_0(input)
        output = self.conv_1(output)
        output = self.conv_2(output)
        output = self.conv_3(output)
        output = self.conv_4(output)
        output = self.conv_5(output)
        output = self.conv_6(output)
        output = output.view(-1, 6 * 6 * 1024)
        output = self.fc1(output)

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