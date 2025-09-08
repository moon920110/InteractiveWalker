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
from matplotlib.gridspec import GridSpec
import cv2
# from drawnow import *
from matplotlib.animation import FuncAnimation


def animate(i, x, ax1, ax2, out_, label_, video):
    ax1.imshow(video.reshape(480, 640, 3))
    ax1.set_title("video")

    ax2.plot(x, out_)
    ax2.plot(x, label_)


def visualize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    Cols = ['LT TIB.ANT. (uV)', 'LT LAT. GASTRO (uV)', 'LT VLO (uV)',
            'LT SEMITEND. (uV)', 'LT GLUT. MAX. (uV)', 'LT RECT.ABDOM.UP. (uV)',
            'RT TIB.ANT. (uV)', 'RT LAT. GASTRO (uV)', 'RT VLO (uV)',
            'RT SEMITEND. (uV)', 'RT GLUT. MAX. (uV)', 'RT RECT.ABDOM.UP. (uV)']

    test_list = ['PSY', 'AMS', 'HTG', 'KHM']

    test_dataset = ExerciseDataset_viz(args.data_path, args.train_exercise, args.window_size, test_list, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if device.type == 'cuda':
        model = CNN_encoder_decoder(args.window_size).cuda(0)
    else: model = CNN_encoder_decoder(args.window_size)

    model.load_state_dict(torch.load(args.test_model_name))
    model.eval()
    x = [i for i in range(300)]
    output_list = []
    label_list = []
    video_list = []
    data_list = []

    gs = GridSpec(24, 10)
    fig = plt.figure(figsize=(16, 9))
    axv = fig.add_subplot(gs[0:8, 0:4])
    axt = fig.add_subplot(gs[9:, 0:4])
    ax0 = fig.add_subplot(gs[1:3, 5:7])
    ax1 = fig.add_subplot(gs[5:7, 5:7])
    ax2 = fig.add_subplot(gs[9:11, 5:7])
    ax3 = fig.add_subplot(gs[13:15, 5:7])
    ax4 = fig.add_subplot(gs[17:19, 5:7])
    ax5 = fig.add_subplot(gs[21:23, 5:7])
    ax6 = fig.add_subplot(gs[1:3, 8:])
    ax7 = fig.add_subplot(gs[5:7, 8:])
    ax8 = fig.add_subplot(gs[9:11, 8:])
    ax9 = fig.add_subplot(gs[13:15, 8:])
    ax10 = fig.add_subplot(gs[17:19, 8:])
    ax11 = fig.add_subplot(gs[21:23, 8:])

    axs = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11]
    axs_labels = ['LT TIB.ANT', 'LT LAT. GASTRO', 'LT VLO',
            'LT SEMITEND', 'LT GLUT', 'LT RECT.ABDOM',
            'RT TIB.ANT', 'RT LAT. GASTRO', 'RT VLO',
            'RT SEMITEND', 'RT GLUT', 'RT RECT.ABDOM']

    plt.show(block=False)


    for i, (data, label, video) in enumerate(test_loader):
        if device.type == 'cuda':
            # data, label = data.cuda(0), label.reshape(-1, 1).cuda(0) # label.reshape(-1, 1) for only 1 muscle tracking
            data, label = data.cuda(0), label.reshape(-1, 12).cuda(0)  # for 12 muscle
        outputs = model(data).detach()
        output_list.append(np.array(outputs.cpu()).reshape(-1))
        label_list.append(np.array(label.cpu()).reshape(-1))
        video_list.append(video)
        data_list.append(data.cpu())
        print(i)
        if i < 300:
            continue
        if i % 10 == 0:
            output_list = output_list[-300:]
            label_list = label_list[-300:]
            print(len(output_list))

            out_ = np.array(output_list)[:, 4]
            label_ = np.array(label_list)[:, 4]
            plt.cla()
            axv.imshow(video.reshape(480, 640, 3))
            axv.set_title("video")
            axv.axis('off')
            temp = np.clip(np.array(data[0, -1, :, :].cpu())/1500*255, 0, 255).reshape(64, 64, 1)
            axt.imshow(np.concatenate((temp, temp, temp), axis=2).astype(np.uint8))
            axt.set_title('tactile')
            for i, ax in enumerate(axs):
                ax.cla()
                ax.plot(x, np.array(output_list)[:, i])
                ax.plot(x, np.array(label_list)[:, i])
                ax.set_ylim([0, 1])
                ax.set_title(axs_labels[i])
            # ax2.plot(x, out_)
            # ax2.plot(x, label_)
            # ax2.set_ylim([0, 1])
            plt.draw()
            plt.pause(0.0000001)

    # viz_result = [output_list, label_list, video_list, data_list]
    # with open('../visualize_result/' + 'viz_result_0305', 'wb') as f:
    #     pickle.dump(viz_result, f)

def save_visualize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for file_name in os.listdir(args.save_viz):
        print(file_name)
        if file_name == '.DS_Store': continue
        path = args.save_viz + '/' + file_name
        with open(path, 'rb') as f:
            viz_file = pickle.load(f)
            outputs = viz_file[0]
            labels = viz_file[1]
            videos = viz_file[2]
            tactiles = viz_file[3]
        test_list = ['PSY', 'AMS', 'HTG', 'KHM']


        x = [i for i in range(300)]
        output_list = np.array(outputs[: 300])[:, 0]
        label_list = np.array(labels[: 300])[:, 0]

        gs = GridSpec(24, 10)
        fig = plt.figure(figsize=(16, 9))
        axv = fig.add_subplot(gs[0:8, 0:4])
        axt = fig.add_subplot(gs[9:, 0:4])
        ax0 = fig.add_subplot(gs[1:3, 5:7])
        ax1 = fig.add_subplot(gs[5:7, 5:7])
        ax2 = fig.add_subplot(gs[9:11, 5:7])
        ax3 = fig.add_subplot(gs[13:15, 5:7])
        ax4 = fig.add_subplot(gs[17:19, 5:7])
        ax5 = fig.add_subplot(gs[21:23, 5:7])
        ax6 = fig.add_subplot(gs[1:3, 8:])
        ax7 = fig.add_subplot(gs[5:7, 8:])
        ax8 = fig.add_subplot(gs[9:11, 8:])
        ax9 = fig.add_subplot(gs[13:15, 8:])
        ax10 = fig.add_subplot(gs[17:19, 8:])
        ax11 = fig.add_subplot(gs[21:23, 8:])
        axv.set_title("video")
        axv.axis('off')
        axt.set_title('tactile')

        viz_axv = axv.imshow(np.array(videos[0]).reshape(480, 640, 3))
        temp = np.clip(np.array(tactiles[0 + 300][0, -1, :, :].cpu()) / 1500 * 255, 0, 255).reshape(64, 64, 1)
        viz_axt = axt.imshow(np.concatenate((temp, temp, temp), axis=2).astype(np.uint8))
        ax0p, = ax0.plot(x, output_list)
        ax1p, = ax1.plot(x, output_list)
        ax2p, = ax2.plot(x, output_list)
        ax3p, = ax3.plot(x, output_list)
        ax4p, = ax4.plot(x, output_list)
        ax5p, = ax5.plot(x, output_list)
        ax6p, = ax6.plot(x, output_list)
        ax7p, = ax7.plot(x, output_list)
        ax8p, = ax8.plot(x, output_list)
        ax9p, = ax9.plot(x, output_list)
        ax10p, = ax10.plot(x, output_list)
        ax11p, = ax11.plot(x, output_list)
        ax0l, = ax0.plot(x, label_list)
        ax1l, = ax1.plot(x, label_list)
        ax2l, = ax2.plot(x, label_list)
        ax3l, = ax3.plot(x, label_list)
        ax4l, = ax4.plot(x, label_list)
        ax5l, = ax5.plot(x, label_list)
        ax6l, = ax6.plot(x, label_list)
        ax7l, = ax7.plot(x, label_list)
        ax8l, = ax8.plot(x, label_list)
        ax9l, = ax9.plot(x, label_list)
        ax10l, = ax10.plot(x, label_list)
        ax11l, = ax11.plot(x, label_list)

        axs = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11]
        axs_labels = ['LT TIB.ANT', 'LT LAT. GASTRO', 'LT VLO',
                      'LT SEMITEND', 'LT GLUT', 'LT RECT.ABDOM',
                      'RT TIB.ANT', 'RT LAT. GASTRO', 'RT VLO',
                      'RT SEMITEND', 'RT GLUT', 'RT RECT.ABDOM']
        for i, ax in enumerate(axs):
            ax.set_ylim([0, 1])
            ax.set_title(axs_labels[i])

        axsp = [ax0p, ax1p, ax2p, ax3p, ax4p, ax5p, ax6p, ax7p, ax8p, ax9p, ax10p, ax11p]
        axsl = [ax0l, ax1l, ax2l, ax3l, ax4l, ax5l, ax6l, ax7l, ax8l, ax9l, ax10l, ax11l]
        axs_labels = ['LT TIB.ANT', 'LT LAT. GASTRO', 'LT VLO',
                'LT SEMITEND', 'LT GLUT', 'LT RECT.ABDOM',
                'RT TIB.ANT', 'RT LAT. GASTRO', 'RT VLO',
                'RT SEMITEND', 'RT GLUT', 'RT RECT.ABDOM']

        def animate(i):
            output_list = outputs[500 + i: 800 + i]
            label_list = labels[500 + i: 800 + i]
            viz_axv.set_array(videos[800 + i].reshape(480, 640, 3))
            temp = np.clip(np.array(tactiles[i + 800][0, -1, :, :].cpu()) / 1500 * 255, 0, 255).reshape(64, 64, 1)
            viz_axt.set_array(np.concatenate((temp, temp, temp), axis=2).astype(np.uint8))
            for j, ax in enumerate(axsp):
                ax.set_data(x, np.array(output_list)[:, j])
            for j, ax in enumerate(axsl):
                ax.set_data(x, np.array(label_list)[:, j])


        ani = FuncAnimation(fig, animate, 500, interval=1)
        # plt.show()

        ani.save('./animation_'+ file_name +'.gif', fps=100)
        print(file_name, 'animation has been saved')

def calculate_error(args):
    outputs = []
    labels = []
    for filename in os.listdir(args.save_viz_folder):
        if 'crunch' not in filename: continue
        path = args.save_viz_folder + filename
        with open(path, 'rb') as f:
            viz_file = pickle.load(f)
            outputs.append(viz_file[0])
            labels.append(viz_file[1])

    print('check')
    test_list = ['PSY', 'AMS', 'HTG', 'KHM']





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tactile2EMG code')
    parser.add_argument('--isTest', type=bool, default=False, help='Test or not')
    # parser.add_argument('--test_model_name', type=str, default='../models/model0302_cnn_smooth_win20_12muscle_continuefrom30_60.pt', help='name of model')
    parser.add_argument('--test_model_name', type=str,
                        default='../models/model0304_all_cnn_win20_12muscle_continuefrom0_2.pt',
                        help='name of model')
    parser.add_argument('--data_path', type=str, default='../data/temp/', help='Experiment path')
    parser.add_argument('--epoch', type=int, default=1, help='total epoch')
    parser.add_argument('--window_size', type=int, default=20, help='window_size')
    parser.add_argument('--batch_size', type=int, default=1, help='total epoch')
    parser.add_argument('--save_path', type=str, default='../models/', help='save path')
    parser.add_argument('--model_name', type=str, default='test', help='name of model')
    parser.add_argument('--train_exercise', type=str, default='all', help='all, crunch, squat, lunge')
    parser.add_argument('--save_viz', type=str, default='../visualize_result3/', help='saved visualize file')
    parser.add_argument('--save_viz_folder', type=str, default='../visualize_result2/',
                        help='saved visualize file')
    # parser.add_argument('--show_len', type=int, default=, help='window_size')


    args = parser.parse_args()

    # visualize(args)
    save_visualize(args)
    # calculate_error(args)