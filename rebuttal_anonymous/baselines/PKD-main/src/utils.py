import os
import os.path as osp
import random
import numpy as np
import torch


def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def reset(value):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)


def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)


def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        st_ = "{}_{}_".format(name, val)
        st += st_

    return st[:-1]


def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals


# def compute_accuracy(preds, labels, running_train_mask, train_mask, val_mask, test_mask, device):
def compute_accuracy(preds, labels, running_train_mask, train_mask, val_mask, test_mask, device):
    for i in range(len(train_mask)):
        if running_train_mask[i] == torch.tensor(True, dtype=torch.bool):
            test_mask[i] = torch.tensor(False, dtype=torch.bool).to(device)
            val_mask[i] = torch.tensor(False, dtype=torch.bool).to(device)
            # test_mask[i] = torch.tensor(False, dtype=torch.bool)
            # val_mask[i] = torch.tensor(False, dtype=torch.bool)

    train_preds = preds[train_mask]
    val_preds = preds[val_mask]
    test_preds = preds[test_mask]

    train_acc = (torch.sum(train_preds == labels[train_mask])).float() / ((labels[train_mask].shape[0]))
    val_acc = (torch.sum(val_preds == labels[val_mask])).float() / ((labels[val_mask].shape[0]))
    test_acc = (torch.sum(test_preds == labels[test_mask])).float() / ((labels[test_mask].shape[0]))

    train_acc = train_acc * 100
    val_acc = val_acc * 100
    test_acc = test_acc * 100

    return train_acc, val_acc, test_acc


def masking(fold, data, label_rate=0.01):
    if label_rate == 1:
        train_mask = data.train_mask1[fold];
        val_mask = data.val_mask1[fold];
        test_mask = data.test_mask1[fold]
    elif label_rate == 2:
        train_mask = data.train_mask2[fold];
        val_mask = data.val_mask2[fold];
        test_mask = data.test_mask2[fold]
    elif label_rate == 3:
        train_mask = data.train_mask3[fold];
        val_mask = data.val_mask3[fold];
        test_mask = data.test_mask3[fold]
    elif label_rate == 4:
        train_mask = data.train_mask4[fold];
        val_mask = data.val_mask4[fold];
        test_mask = data.test_mask4[fold]
    elif label_rate == 5:
        train_mask = data.train_mask5[fold];
        val_mask = data.val_mask5[fold];
        test_mask = data.test_mask5[fold]
    elif label_rate == 6:
        train_mask = data.train_mask6[fold];
        val_mask = data.val_mask6[fold];
        test_mask = data.test_mask6[fold]
    elif label_rate == 7:
        train_mask = data.train_mask7[fold];
        val_mask = data.val_mask7[fold];
        test_mask = data.test_mask7[fold]
    elif label_rate == 8:
        train_mask = data.train_mask8[fold];
        val_mask = data.val_mask8[fold];
        test_mask = data.test_mask8[fold]
    elif label_rate == 9:
        train_mask = data.train_mask9[fold];
        val_mask = data.val_mask9[fold];
        test_mask = data.test_mask9[fold]
    elif label_rate == 10:
        train_mask = data.train_mask10[fold];
        val_mask = data.val_mask10[fold];
        test_mask = data.test_mask10[fold]
    elif label_rate == 11:
        train_mask = data.train_mask11[fold];
        val_mask = data.val_mask11[fold];
        test_mask = data.test_mask11[fold]

    return train_mask, val_mask, test_mask


def compute_representation(net, data, device):
    net.eval()
    reps = []

    data = data.to(device)
    with torch.no_grad():
        reps.append(net(data))

    reps = torch.cat(reps, dim=0)

    return reps