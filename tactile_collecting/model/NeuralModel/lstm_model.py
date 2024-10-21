import torch.nn as nn
import torch as tr
from model.const import CLASS_NUM

class LSTMCNN_hc(nn.Module):

    def __init__(self, windowSize):
        super(LSTMCNN_hc, self).__init__()

        self.conv_0 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2))

        self.conv_1 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4))

        self.conv_2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8))

        self.fc_1 = nn.Sequential(
            nn.Linear(32768, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )

        self.lstm = nn.LSTM(input_size=512,
                            hidden_size=512,
                            num_layers=1)

        self.windowSize = windowSize

        self.fc_2 = nn.Sequential(
            nn.Linear(512 * self.windowSize, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, CLASS_NUM),
            nn.Softmax()
        )

    def init_hidden(self, batch_size):
        return (tr.autograd.Variable(tr.zeros(1, batch_size, 512)).cuda(),
                tr.autograd.Variable(tr.zeros(1, batch_size, 512)).cuda())

    def forward(self, input):
        bs = input.size(0)
        length = input.size(1)
        outputs = tr.autograd.Variable(tr.zeros(bs, length, 512)).cuda()
        for i in range(length):
            # input[:, i] = (bs, h, w)
            # unsqueeze(1) -> (bs, 1, h, w)
            output = self.conv_0(input[:, i].unsqueeze(1))
            output = self.conv_1(output)
            output = self.conv_2(output)
            output = output.view(output.size(0), -1)
            output = self.fc_1(output)
            outputs[:, i] = output
        outputs,_ = self.lstm(outputs)
        outputs = outputs.reshape(-1, 512 * self.windowSize)
        outputs = self.fc_2(outputs)
        return outputs


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_0 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2))
        # 4,64,64

        self.conv_1 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(kernel_size=2)) # 48 * 48
        # 8*32*32


        self.conv_2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2))
        #16*16*16


        self.conv_3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2)) # 24 * 24
        #32*8*8

        self.flatten = nn.Flatten()

        self.conv_4 = nn.Sequential(
            nn.Linear(1024, 256),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU()
        )


        # self.conv_4 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=(3,3),padding=1),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm2d(512))
        # #512*16*16
        # #512*8*8
        #
        # self.conv_5 = nn.Sequential(
        #     nn.Conv2d(512, 1024, kernel_size=(5,5)),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm2d(1024))

    def forward(self, input):
        output = self.conv_0(input)
        output = self.conv_1(output)
        output = self.conv_2(output)
        output = self.conv_3(output)
        output = self.flatten(output)
        output = self.conv_4(output)
        # output = self.conv_4(output)
        # output = self.conv_5(output)
        # output = self.conv_6(output)
        output = output.view(-1, 256)
        return output


class LSTMCNN_yh(nn.Module):
    def __init__(self):
        super(LSTMCNN_yh, self).__init__()
        self.cnn = CNN()
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=64,
            num_layers=1,
            batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(64, CLASS_NUM),
            nn.Softmax()
        )

    def forward(self, input):
        input = input.unsqueeze(2)
        batch_size, timestamps, C, H, W = input.size()
        cnn_in = input.view(batch_size * timestamps, C, H, W)
        cnn_out = self.cnn(cnn_in)
        rnn_in = cnn_out.view(batch_size, timestamps, -1)
        rnn_out, (h_n, h_c) = self.rnn(rnn_in)
        rnn_out2 = self.linear(rnn_out[:,-1,:])

        return rnn_out2
