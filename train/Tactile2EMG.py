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
from Tactile2EMG_models import *
import sympy as sy
import scipy
import dtw

# def moving_avarage_smoothing(X,k):
# 	S = np.zeros(X.shape[0])
# 	for t in range(X.shape[0]):
# 		if t < k:
# 			S[t] = np.mean(X[:t+1])
# 		else:
# 			S[t] = np.sum(X[t-k:t])/k
# 	return S

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    test_list = ['tes']

    train_dataset = ExerciseDataset(args.data_path, args.train_exercise, args.window_size, test_list, is_test=False)
    test_dataset = ExerciseDataset(args.data_path, args.train_exercise, args.window_size, test_list, is_test=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if device.type == 'cuda':
        # model = CNN_encoder_decoder(args.window_size).cuda(0)
        model = CNN_encoder_decoder0906(args.window_size).cuda(0)
    else:
        # model = CNN_encoder_decoder(args.window_size)
        model = CNN_encoder_decoder0906(args.window_size)
        # model = CNN_encoder_decoder_classifier(args.window_size)
    outputs_list = []


    # Define loss function and optimizer
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001) # 0.0003

    # Training loop
    num_epochs = args.epoch
    print('training start')
    for epoch in range(num_epochs):
        loss_mean = 0
        test_loss_mean = 0
        mean_r2 = 0
        mean_r2_eval = 0
        model.train()
        outputs_t = []
        labels_t = []
        outputs_v = []
        labels_v = []
        model.train()
        train_true_list = []
        train_tot_num_list = []
        for i, (data, label) in enumerate(train_loader):
            label = label.reshape(-1, 1)
            optimizer.zero_grad()
            outputs = model(data)
            # outputs = outputs.squeeze(1)
            loss = criterion(outputs, label)
            # loss = (abs(outputs - label) * abs(outputs)).mean() * 1000
            loss.backward()
            optimizer.step()
            loss_mean += loss.cpu().detach()/len(train_loader)
            train_tot_num_list.append(len(outputs.detach()))
            train_true_list.append((np.argmax(outputs.detach(), axis=1) == np.argmax(label, axis=1)).sum())
            if epoch % 10 == 0:
                if i == 0:
                    outputs_t = outputs.cpu().detach().numpy()
                    labels_t = label.cpu().detach().numpy()
                else:
                    outputs_t = numpy.concatenate((outputs_t, outputs.cpu().detach().numpy()), axis=0)
                    labels_t = numpy.concatenate((labels_t, label.cpu().detach().numpy()), axis=0)
            mean_r2 += r2_score(outputs.cpu().detach().numpy(), label.cpu().detach().numpy()) / len(train_loader)

        if args.isTest == True:
            model.load_state_dict(torch.load(args.test_model_name))
        model.eval()
        true_list = []
        tot_num_list = []
        for i, (data,label) in enumerate(test_loader):
            label = label.reshape(-1, 1)
            if device.type == 'cuda':
                # data, label = data.cuda(0), label.reshape(-1, 1).cuda(0) # label.reshape(-1, 1) for only 1 muscle tracking
                data, label = data.cuda(0), label.reshape(-1, 12).cuda(0)  # for 12 muscle
            outputs = model(data).detach()
            # outputs = outputs.squeeze(1)
            loss = criterion(outputs, label)
            # loss = (abs(outputs - label) * abs(outputs)).mean() * 1000
            # if args.isTest == True:
            #     for i in range(len(outputs) - 5):
            #         if i != 0:
            #             outputs[-i] = np.mean(outputs[-(i + 5):-i])

            tot_num_list.append(len(outputs))
            true_list.append((np.argmax(outputs, axis=1) == np.argmax(label, axis=1)).sum())
            test_loss_mean += loss.cpu().detach().numpy() / len(test_loader)
            mean_r2_eval += r2_score(outputs.cpu().detach().numpy(), label.cpu().detach().numpy()) / len(train_loader)
            if epoch % 10 == 0:
                if i == 0:
                    outputs_v = outputs.cpu().detach().numpy()
                    labels_v = label.cpu().detach().numpy()
                else:
                    outputs_v = numpy.concatenate((outputs_v, outputs.cpu().detach().numpy()), axis=0)
                    labels_v = numpy.concatenate((labels_v, label.cpu().detach().numpy()), axis=0)
            mean_r2 += r2_score(outputs.cpu().detach().numpy(), label.cpu().detach().numpy()) / len(train_loader)
        if epoch % 10 == 0: #epoch == num_epochs - 1:
            plt.clf()
            x_t = [k for k in range(0, 300)]
            x_v = [k for k in range(0, 300)]
            plt.figure(figsize=(8,6))
            for i in range(len(outputs_v)-5):
                if i != 0:
                    outputs_v[-i] = np.mean(outputs_v[-(i+5):-i])
            # labels_v = moving_avarage_smoothing(labels_v, 10)
            # outputs_v = moving_avarage_smoothing(outputs_v, 10)
            plt.plot(x_v, outputs_v[1100:1400], label='predicted') # for 12 muscle 480:780
            plt.plot(x_v, labels_v[1100:1400], label='original') # for 12 muscle
            plt.legend()
            # plt.show()
            plt.savefig('../results/' + args.model_name + '_' + str(epoch) + '.png')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss_mean:.7f}, Test Loss: {test_loss_mean:.7f}, train_acc: {sum(train_true_list)/sum(train_tot_num_list)},test_acc: {sum(true_list)/sum(tot_num_list)}')
        # labels_v = moving_avarage_smoothing(labels_v, 20)
        # outputs_v = moving_avarage_smoothing(outputs_v, 10)
        if args.isTest == True:
            print(outputs_v - labels_v)
            print(f'acc diff mean: {abs(outputs_v - labels_v).mean()}, std: {abs(outputs_v - labels_v).std()}')
            zero_list = [0]
            for i in range(len(labels_v)):
                if labels_v[zero_list[-1]] >= 0:
                    if labels_v[i] < 0: zero_list.append(i)
                elif labels_v[zero_list[-1]] < 0:
                    if labels_v[i] >= 0: zero_list.append(i)
                else: pass
            speed_gap_list = []
            # for i in range(len(zero_list)-1):
            for i in range(int(len(labels_v)/12) - 1):
                if i == 35:
                    print('hi')
                # speed_gap = abs(labels_v[zero_list[i]:zero_list[i+1]].sum() - outputs_v[zero_list[i]:zero_list[i+1]].sum())/len(labels_v[zero_list[i]:zero_list[i+1]])
                speed_gap = abs(labels_v[i*12:(i+1)*12].sum() - outputs_v[i*12:(i+1)*12].sum()) / 12
                speed_gap_list.append(speed_gap)
            print(dtw.dtw(labels_v,outputs_v, keep_internals=True).distance)
            print('speed_gap diff:, ')

        if epoch % 10 == 0:
            torch.save(model.state_dict(), args.save_path + args.model_name + '_' + str(epoch) +'.pt')
        torch.save(model.state_dict(), args.save_path + args.model_name + str(num_epochs) +'.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tactile2EMG code')
    parser.add_argument('--isTest', type=bool, default=True, help='Test or not')
    parser.add_argument('--test_model_name', type=str, default='../models/model0906_acc_estimator_970.pt', help='name of model')
    # parser.add_argument('--test_model_name', type=str, default='../models/model0402_acc_estimator_940.pt', help='name of model')
    parser.add_argument('--data_path', type=str, default='../data/smoothed_data_cut_labels_splitTest/', help='Experiment path')
    parser.add_argument('--epoch', type=int, default=1000, help='total epoch')
    parser.add_argument('--window_size', type=int, default=10, help='window_size')
    parser.add_argument('--batch_size', type=int, default=100, help='total epoch')
    parser.add_argument('--save_path', type=str, default='../models/', help='save path')
    parser.add_argument('--model_name', type=str, default='model0906_acc_estimator', help='name of model')
    parser.add_argument('--train_exercise', type=str, default='all', help='all, crunch, squat, lunge')
    parser.add_argument('--start_exercise', type=int, default=1, help='when clap on right camera')
    parser.add_argument('--calibrate', type=bool, default=False, help='Set true if you calibrate')

    args = parser.parse_args()

    train(args)