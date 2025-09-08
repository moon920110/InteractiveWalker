from model.networks import *
from model.lstm_model import *
from model.dataloader import PressureDataset
import torch
from tensorboardX import SummaryWriter
import time
import argparse
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from model.const import DATA_PATHS, TEST_DATA_PATHS
import torch.nn.functional as F

def print_square(dictionary):
    for key in dictionary.keys():
        # if "float" in str(type(dictionary[key])):
        #     newval = round(float(dictionary[key]), 4)
        #     dictionary[key] = newval

        if None == dictionary[key]:
            dictionary[key] = "None"

    front_lens = []
    back_lens = []
    for key in dictionary.keys():
        front_lens.append(len(key))
        back_lens.append(len(str(dictionary[key])))
    front_len = max(front_lens)
    back_len = max(back_lens)

    strings = []
    for key in dictionary.keys():
        string = "| {0:<{2}} | {1:<{3}} |".format(key, dictionary[key], front_len, back_len)
        strings.append(string)

    max_len = max([len(i) for i in strings])
    print("-"*max_len)
    for string in strings:
        print(string)
    print("-" * max_len)

def run_epoch(model, optimizer, loader, batch_size, cuda, cuda_idx, step=True):
    length = len(loader)
    acc = 0
    regression_loss_item = 0
    class_loss_item = 0
    total_loss_item = 0

    for i, (x, y_value, y_class) in enumerate(loader):
        if cuda:
            x, y_value, y_class = x.cuda(cuda_idx), y_value.cuda(cuda_idx), y_class.cuda(cuda_idx)
        pred_value, pred_class = model(x)
        regression_loss = F.mse_loss(pred_value, y_value)

        if len(y_class.size()) > 1: y_class = y_class.squeeze()
        class_loss = F.cross_entropy(pred_class, y_class)
        total_loss = regression_loss + class_loss

        if step:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        acc += (torch.argmax(pred_class, dim=1) == y_class).float().sum()
        regression_loss_item += regression_loss.item()
        class_loss_item += class_loss.item()
        total_loss_item += total_loss.item()
    acc = acc * 100 / (length * batch_size)
    regression_loss_item /= length
    class_loss_item /= length
    total_loss_item /= length

    history = {
        "regression_loss": regression_loss_item,
        "classification_loss": class_loss_item,
        "accuracy": acc,
        "total_loss": total_loss_item,
    }
    return history


def run_epoch_v2(model, optimizer, loader, batch_size, cuda, cuda_idx, step=True):
    length = len(loader)

    speed_loss_item = 0
    motion_loss_item = 0
    angle_loss_item = 0
    motion_acc = 0
    angle_acc = 0
    total_loss_item = 0

    for i, (x, motion_class) in enumerate(loader):
        if cuda:
            x, motion_class, = \
                x.cuda(cuda_idx), motion_class.cuda(cuda_idx)
        if len(motion_class.size()) > 1: motion_class = motion_class.squeeze()

        pred_motion = model(x)

        motion_loss = F.cross_entropy(pred_motion, motion_class)

        if step:
            optimizer.zero_grad()
            motion_loss.backward()
            optimizer.step()

        motion_acc += (torch.argmax(pred_motion, dim=1) == motion_class).float().sum()
        motion_loss_item += motion_loss.item()
        total_loss_item += motion_loss.item()

    motion_acc = motion_acc * 100 / (length * batch_size)
    motion_loss_item /= length

    history = {
        "motion_accuracy": motion_acc,
        "motion_loss": motion_loss_item,
    }
    return history



def main(args):
    data_path = DATA_PATHS
    test_data_path = TEST_DATA_PATHS
    #valid_path = args.valid_path
    log_save_path = os.path.join(
        args.log_save_dir,
        f"model_{args.id_string}_{args.model_type}_lr{args.learning_rate}_w_size{args.window_size}_{str(int(time.time()))[4:]}/"
    )
    model_type = args.model_type
    batch_size = args.batch_size
    window_size = args.window_size
    #REGRESS_NUM = args.REGRESS_NUM
    learning_rate = args.learning_rate
    cuda = args.cuda
    cuda_idx = args.cuda_idx
    max_epoch = args.max_epoch
    #train_max_len = args.train_max_len
    #valid_max_len = args.valid_max_len

    logger = SummaryWriter(log_save_path)

    args_print = {}
    for arg in vars(args):
        logger.add_text(arg, str(getattr(args, arg)), 0)
        args_print[arg] = getattr(args, arg)
    args_print["log_save_path"] = log_save_path

    print_square(args_print)

    split_ratio = 0.8
    dataset = PressureDataset(data_path, window_size)
    test_dataset = PressureDataset(test_data_path, window_size)

    train_size = int(split_ratio * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    '''
    train_loader = Dataloader(train_path, batch_size, window_size, max_len=train_max_len)
    vaild_loader = Dataloader(valid_path, batch_size, window_size, max_len=valid_max_len)
    '''

    if model_type.lower() == "conv_2d":
        #model = vanilla_conv2d(window_size)
        model = vanilla_conv2d_v2(window_size)
    elif model_type.lower() == "conv_3d":
        model = vanilla_conv3d(window_size)
    elif model_type.lower() == 'lstm_hc':
        model = LSTMCNN_hc(window_size)
    elif model_type.lower() == 'lstm_yh':
        model = LSTMCNN_yh()
    else:
        raise ValueError


    if cuda: model.cuda(cuda_idx)
    model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(max_epoch):

        start_time = time.time()

        model.train()
        train_history = run_epoch_v2(model, optimizer, train_loader, batch_size, cuda, cuda_idx, step=True)

        model.eval()
        valid_history = run_epoch_v2(model, optimizer, valid_loader, batch_size, cuda, cuda_idx, step=False)
        test_history = run_epoch_v2(model, optimizer, test_loader, batch_size, cuda, cuda_idx, step=False)

        history = {
            "epoch": epoch,
            "time/epoch": time.time() - start_time
        }

        for key in train_history.keys():
            history["train/"+key] = train_history[key]
        for key in valid_history.keys():
            history["valid/"+key] = valid_history[key]
        for key in test_history.keys():
            history["test/"+key] = test_history[key]


        print_square(history)

        for key in history.keys():
            logger.add_scalar(key, history[key], epoch)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(log_save_path, f"model_{epoch}.pth"))
        #torch.save(model.state_dict(), os.path.join(log_save_path, f"model.pth"))
    torch.save(model.state_dict(), os.path.join(log_save_path, f"model_{epoch}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--id-string", type=str, default="")
    parser.add_argument("--model-type", type=str, default="lstm_yh")
    #parser.add_argument("--train-path", type=str, default="./data/train/")
    #parser.add_argument("--valid-path", type=str, default="./data/val/")
    parser.add_argument("--log-save-dir", type=str, default=f"./saved/")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=20)
    #parser.add_argument("--class-num", type=int, default=3)
    #parser.add_argument("--regress-num", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.000001)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument("--max-epoch", type=int, default=50)
    #parser.add_argument("--train-max-len", type=int, default=None)
    #parser.add_argument("--valid-max-len", type=int, default=None)
    args = parser.parse_args()

    main(args)

    '''
    learning_rates = [0.00005]
    batch_sizes = [1024]
    windows_sizes = [20]
    for lr in learning_rates:
        for w_size in windows_sizes:
            for b_size in batch_sizes:
                args.id_string = f"master_3d_lr{lr}_w_size{w_size}_b_size{b_size}"
                args.learning_rate = lr
                args.window_size = w_size
                args.batch_size = b_size
                main(args)
    '''






