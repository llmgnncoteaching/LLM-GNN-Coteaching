import torch
import numpy as np
import os.path as osp
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon
from src.utils import create_dirs
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB, Actor

# from read import get_raw_text_webkb

def get_data_tag(dataset_name):
    # data, _ = get_raw_text_webkb(dataset_name, use_text=True, seed=0)
    dataset = torch.load('/share/home/u20526/wx/walk_of_thoughts/tag_dataset/{}.pt'.format(dataset_name))
    nodes = torch.tensor(np.arange(dataset.num_nodes), dtype=torch.long)

    create_masks(data=dataset)

    data = Data(nodes=nodes, edge_index=dataset.edge_index, edge_attr=dataset.edge_attr, x=dataset.x, y=dataset.y,
                train_mask1=dataset.train_mask1, val_mask1=dataset.val_mask1, test_mask1=dataset.test_mask1,
                train_mask2=dataset.train_mask2, val_mask2=dataset.val_mask2, test_mask2=dataset.test_mask2,
                train_mask3=dataset.train_mask3, val_mask3=dataset.val_mask3, test_mask3=dataset.test_mask3,
                train_mask4=dataset.train_mask4, val_mask4=dataset.val_mask4, test_mask4=dataset.test_mask4,
                train_mask5=dataset.train_mask5, val_mask5=dataset.val_mask5, test_mask5=dataset.test_mask5,
                num_nodes=dataset.num_nodes)
    texts = dataset.raw_texts

    return data, texts

def create_masks(data, name='cora'):
    """
    Splits data into training, validation, and test splits in a stratified manner if
    it is not already splitted. Each split is associated with a mask vector, which
    specifies the indices for that split. The data will be modified in-place
    :param data: Data object
    :return: The modified data
    """

    labels = data.y
    mask1 = Random_Nonzero_Masking(data, labels, label_rate=1)
    mask2 = Random_Nonzero_Masking(data, labels, label_rate=2)
    mask3 = Random_Nonzero_Masking(data, labels, label_rate=3)
    mask4 = Random_Nonzero_Masking(data, labels, label_rate=4)
    mask5 = Random_Nonzero_Masking(data, labels, label_rate=5)

    data.train_mask1, data.val_mask1, data.test_mask1 = mask1
    data.train_mask2, data.val_mask2, data.test_mask2 = mask2
    data.train_mask3, data.val_mask3, data.test_mask3 = mask3
    data.train_mask4, data.val_mask4, data.test_mask4 = mask4
    data.train_mask5, data.val_mask5, data.test_mask5 = mask5

    return data

def Random_Nonzero_Masking(data, labels, label_rate):
    train_size = np.unique(labels).shape[0] * label_rate
    eval_size = data.x.size(0) - train_size
    dev_size = int(eval_size * 0.1)

    final_train_mask, final_val_mask, final_test_mask = None, None, None
    cnt = 0

    # 一次性生成全局的随机排列
    labels_numpy = labels.numpy()
    data_index = np.arange(labels_numpy.shape[0])

    while cnt < 20:
        # 按类别采样
        train_index = []
        for i in range(np.unique(labels).shape[0]):
            indices = np.where(labels_numpy == i)[0]

            # 如果该类别的节点数为 0，跳过该类别
            if len(indices) == 0:
                continue

            # 如果该类别节点少于 label_rate，直接将所有节点加入训练集
            if len(indices) >= label_rate:
                train_index.extend(np.random.choice(indices, size=label_rate, replace=False))
            else:
                train_index.extend(indices)

        remaining_index = np.setdiff1d(data_index, train_index)
        np.random.shuffle(remaining_index)

        dev_index = remaining_index[: dev_size]
        test_index = remaining_index[dev_size:]

        train_mask = torch.zeros(data_index.shape[0], dtype=torch.bool)
        dev_mask = torch.zeros(data_index.shape[0], dtype=torch.bool)
        test_mask = torch.zeros(data_index.shape[0], dtype=torch.bool)

        train_mask[train_index] = True
        dev_mask[dev_index] = True
        test_mask[test_index] = True

        # 确保训练集中每个类别都有至少一个节点
        if np.unique(labels_numpy[train_mask]).shape[0] == np.unique(labels).shape[0]:
            cnt += 1
            if final_train_mask is None:
                final_train_mask = train_mask.unsqueeze(0)
                final_val_mask = dev_mask.unsqueeze(0)
                final_test_mask = test_mask.unsqueeze(0)
            else:
                final_train_mask = torch.cat((final_train_mask, train_mask.unsqueeze(0)), dim=0)
                final_val_mask = torch.cat((final_val_mask, dev_mask.unsqueeze(0)), dim=0)
                final_test_mask = torch.cat((final_test_mask, test_mask.unsqueeze(0)), dim=0)

    return final_train_mask, final_val_mask, final_test_mask




