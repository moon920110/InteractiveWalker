import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import pandas as pd
from Tactile2pose_dataLoader import *
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import argparse
from Tactile2pose_models import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from threeD_viz_video import *



def remove_small(heatmap, threshold, device):
    z = torch.zeros(heatmap.shape[0], heatmap.shape[1], heatmap.shape[2], heatmap.shape[3], heatmap.shape[4]).to(device)
    heatmap = torch.where(heatmap<threshold, z, heatmap)
    return heatmap

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    Cols = ['LT TIB.ANT. (uV)', 'LT LAT. GASTRO (uV)', 'LT VLO (uV)',
            'LT SEMITEND. (uV)', 'LT GLUT. MAX. (uV)', 'LT RECT.ABDOM.UP. (uV)',
            'RT TIB.ANT. (uV)', 'RT LAT. GASTRO (uV)', 'RT VLO (uV)',
            'RT SEMITEND. (uV)', 'RT GLUT. MAX. (uV)', 'RT RECT.ABDOM.UP. (uV)']

    test_list = ['PSY', 'AMS', 'HTG', 'KHM']

    tactile_GT = np.empty((1, 64, 64))
    heatmap_GT = np.empty((1, 12, 16, 16, 16))
    heatmap_pred = np.empty((1, 12, 16, 16, 16))
    keypoint_GT = np.empty((1, 12, 3))
    keypoint_pred = np.empty((1, 12, 3))
    tactile_GT_v = np.empty((1, 64, 64))
    heatmap_GT_v = np.empty((1, 12, 16, 16, 16))
    heatmap_pred_v = np.empty((1, 12, 16, 16, 16))
    keypoint_GT_v = np.empty((1, 12, 3))
    keypoint_pred_v = np.empty((1, 12, 3))
    keypoint_GT_log = np.empty((1, 12, 3))
    keypoint_pred_log = np.empty((1, 12, 3))

    train_dataset = ExerciseDataset(args.data_path, args.train_exercise, args.window_size, test_list, is_test=False)
    test_dataset = ExerciseDataset(args.data_path, args.train_exercise, args.window_size, test_list, is_test=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if device.type == 'cuda':
        model = CNN_intelligent_carpet(args.window_size).cuda(0)
    else: model = CNN_intelligent_carpet(args.window_size)

    if args.train_continue:
        print('train continue from {}'.format(args.continue_model))
        model.load_state_dict(torch.load(args.continue_model))

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5, verbose=True)
    softmax = SpatialSoftmax3D(16, 16, 16, 12)

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
        if args.isTest == False:
            for i, (data, label) in enumerate(train_loader):
                if device.type == 'cuda':
                    # data, label = data.cuda(0), label.reshape(-1, 1).cuda(0) # label.reshape(-1, 1) for only 1 muscle tracking
                    data, label = data.cuda(0), label.cuda(0) # for 12 muscle
                optimizer.zero_grad()
                outputs = model(data)
                outputs_transform = remove_small(outputs, 1e-2, device)
                # loss = criterion(outputs, label)
                loss = torch.mean((outputs_transform - label) ** 2 * (label + 0.5) * 2) * 1000
                loss.backward()
                optimizer.step()
                loss_mean += loss.cpu().detach()/len(train_loader)
                # if epoch % 10 == 0:
                #     if i == 0:
                #         outputs_t = outputs.cpu().detach().numpy()
                #         labels_t = label.cpu().detach().numpy()
                #     else:
                #         outputs_t = numpy.concatenate((outputs_t, outputs.cpu().detach().numpy()), axis=0)
                #         labels_t = numpy.concatenate((labels_t, label.cpu().detach().numpy()), axis=0)
                # mean_r2 += r2_score(outputs.cpu().detach().numpy(), label.cpu().detach().numpy()) / len(train_loader)
        keypoint_GT_v_list = []
        keypoint_pred_v_list = []
        if args.isTest == True:
            model.load_state_dict(torch.load(args.test_model_name))
            ''' for test '''
        model.eval()
        for i, (data,label) in enumerate(test_loader):
            if device.type == 'cuda':
                # data, label = data.cuda(0), label.reshape(-1, 1).cuda(0) # label.reshape(-1, 1) for only 1 muscle tracking
                data, label = data.cuda(0), label.cuda(0)  # for 12 muscle
            outputs = model(data).detach()
            # loss = criterion(outputs, label)
            outputs_transform = remove_small(outputs, 1e-2, device)
            loss = torch.mean((outputs_transform - label) ** 2 * (label + 0.5) * 2) * 1000
            test_loss_mean += loss.cpu().detach().numpy() / len(test_loader)
            if args.isTest:

                outputs = outputs.reshape(-1, 12, 16, 16, 16)
                outputs_transform = remove_small(outputs, 1e-2, device)
                keypoint_out, heatmap_out = softmax(outputs_transform)

                label_transform = remove_small(label, 1e-2, device)
                keypoint_label, heatmap_label = softmax(label_transform)

                keypoint_GT_v = np.concatenate((keypoint_GT_v, np.array(keypoint_label.cpu())), axis=0)
                keypoint_pred_v = np.concatenate((keypoint_pred_v, np.array(keypoint_out.cpu())), axis=0)
                # heatmap_GT_v = np.append(heatmap_GT, label.cpu().data.numpy().reshape(-1, 12, 16, 16, 16), axis=0)
                # heatmap_pred_v = np.append(heatmap_pred, heatmap_out.cpu().data.numpy().reshape(-1, 12, 16, 16, 16),
                #                            axis=0)
                # keypoint_GT_v = np.append(keypoint_GT, keypoint_label.cpu().data.numpy().reshape(-1, 12, 3), axis=0)
                # keypoint_pred_v = np.append(keypoint_pred, keypoint_out.cpu().data.numpy().reshape(-1, 12, 3), axis=0)
                # tactile_GT_v = np.append(tactile_GT, data.cpu().data.numpy().reshape(-1, 64, 64), axis=0)
                # keypoint_GT_v_list = keypoint_GT_v_list.append(label.cpu())
                # keypoint_pred_v_list = keypoint_pred_v_list.append(keypoint_out.cpu())
            # mean_r2_eval += r2_score(outputs.cpu().detach().numpy(), label.cpu().detach().numpy()) / len(train_loader)
            # if epoch % 10 == 0:
            #     if i == 0:
            #         outputs_v = outputs.cpu().detach().numpy()
            #         labels_v = label.cpu().detach().numpy()
            #     else:
            #         outputs_v = numpy.concatenate((outputs_v, outputs.cpu().detach().numpy()), axis=0)
            #         labels_v = numpy.concatenate((labels_v, label.cpu().detach().numpy()), axis=0)
        #     mean_r2 += r2_score(outputs.cpu().detach().numpy(), label.cpu().detach().numpy()) / len(train_loader)
        # if epoch % 10 == 0: #epoch == num_epochs - 1:
        #     plt.clf()
        #     x_t = [k for k in range(0, 300)]
        #     x_v = [k for k in range(0, 300)]
        #     plt.figure(figsize=(8,6))
        #     # plt.plot(x_v, outputs_v.reshape(-1)[:300], label='predicted') # for single muscle
        #     # plt.plot(x_v, labels_v.reshape(-1)[:300], label='original') # for single muscle
        #     plt.plot(x_v, outputs_v[:, 5][:300], label='predicted') # for 12 muscle
        #     plt.plot(x_v, labels_v[:, 5][:300], label='original') # for 12 muscle
        #     plt.legend()
        #     # plt.show()
        #     plt.savefig('../results/' + args.model_name + '_' + str(epoch) + '.png')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss_mean:.7f}, Test Loss: {test_loss_mean:.7f}')
        if args.isTest == False:
            if epoch % 10 == 0:
                torch.save(model.state_dict(), args.save_path + args.model_name + '_' + str(epoch) +'.pt')
            torch.save(model.state_dict(), args.save_path + args.model_name + '_' + str(epoch) + '.pt')
            torch.save(model.state_dict(), args.save_path + args.model_name + str(num_epochs) +'.pt')
        else:
            to_save = [heatmap_GT_v[1:, :, :, :, :], heatmap_pred_v[1:, :, :, :, :],
                       keypoint_GT_v[1:, :, :], keypoint_pred_v[1:, :, :],
                       tactile_GT_v[1:, :, :]]

            print(to_save[0].shape, to_save[1].shape, to_save[2].shape, to_save[3].shape, to_save[4].shape)

            generateVideo(to_save,
                          args.save_path + args.model_name + '_1',
                          heatmap=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tactile2EMG code')
    parser.add_argument('--isTest', type=bool, default=True, help='Test or not')
    parser.add_argument('--train_continue', type=bool, default=True, help='Test or not')
    parser.add_argument('--continue_model', type=str, default='../models_pose/model0229_pose_continue4_18.pt', help='Test or not')
    parser.add_argument('--test_model_name', type=str, default='../models_pose/model0229_pose_continue4_18.pt', help='name of model')
    # parser.add_argument('--data_path', type=str, default='../data/test/', help='Experiment path')
    parser.add_argument('--data_path', type=str, default='../data/training_data_heatmap_video/', help='Experiment path')
    parser.add_argument('--epoch', type=int, default=5000, help='total epoch')
    parser.add_argument('--window_size', type=int, default=20, help='window_size')
    parser.add_argument('--batch_size', type=int, default=10, help='total epoch') # 50
    parser.add_argument('--save_path', type=str, default='../models_pose/', help='save path')
    parser.add_argument('--model_name', type=str, default='model0229_pose_continue18', help='name of model')
    parser.add_argument('--train_exercise', type=str, default='all', help='all, crunch, squat, lunge')

    args = parser.parse_args()

    train(args)