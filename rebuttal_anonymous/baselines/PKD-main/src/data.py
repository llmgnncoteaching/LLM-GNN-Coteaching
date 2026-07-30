import torch
import numpy as np
import os.path as osp
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon
from src.utils import create_dirs
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB, Actor


def decide_config(root, dataset):
    """
    Create a configuration to download datasets
    :param root: A path to a root directory where data will be stored
    :param dataset: The name of the dataset to be downloaded
    :return: A modified root dir, the name of the dataset class, and parameters associated to the class
    """
    dataset = dataset.lower()
    if dataset == 'cora' or dataset == 'citeseer' or dataset == "pubmed":
        root = osp.join(root, "pyg", "planetoid")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Planetoid, "src": "pyg"}
    elif dataset == "computers":
        dataset = "Computers"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Amazon, "src": "pyg"}
    elif dataset == "photo":
        dataset = "Photo"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Amazon, "src": "pyg"}
    elif dataset == "squirrel":
        dataset = "squirrel"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": WikipediaNetwork, "src": "pyg"}
    elif dataset == "chameleon":
        dataset = "chameleon"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": WikipediaNetwork, "src": "pyg"}
    elif dataset == "texas":
        dataset = "texas"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": WebKB, "src": "pyg"}
    elif dataset == "wisconsin":
        dataset = "wisconsin"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": WebKB, "src": "pyg"}
    elif dataset == "cornell":
        dataset = "cornell"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": WebKB, "src": "pyg"}
    elif dataset == "actor":
        dataset = "actor"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root},
                  "name": dataset, "class": Actor, "src": "pyg"}
    else:
        raise Exception(
            f"Unknown dataset name {dataset}, name has to be one of the following 'cora', 'citeseer', 'pubmed', 'photo', 'computers'")
    return params

def download_pyg_data(config):
    """
    Downloads a dataset from the PyTorch Geometric library
    :param config: A dict containing info on the dataset to be downloaded
    :return: A tuple containing (root directory, dataset name, data directory)
    """
    leaf_dir = config["kwargs"]["root"].split("/")[-1].strip()
    data_dir = osp.join(config["kwargs"]["root"], "" if config["name"] == leaf_dir else config["name"])
    dst_path = osp.join(data_dir, "raw", "data.pt")
    mask_path = osp.join(data_dir, "processed")
    # dst_path = osp.join(data_dir, "geom_gcn/raw", "data.pt")
    # mask_path = osp.join(data_dir, "geom_gcn/processed")
    if not osp.exists(mask_path):
        DatasetClass = config["class"]
        dataset = DatasetClass(**config["kwargs"], transform = T.NormalizeFeatures())
        data = dataset[0]
        create_masks(data=data, name=config["name"])
        torch.save((data, dataset.slices), dst_path)
    
    return config["kwargs"]["root"], config["name"], data_dir


def download_data(root, dataset):
    """
    Download data from different repositories. Currently only PyTorch Geometric is supported

    :param root: The root directory of the dataset
    :param name: The name of the dataset
    :return:
    """
    config = decide_config(root=root, dataset=dataset)
    if config["src"] == "pyg":
        return download_pyg_data(config)


class Dataset(InMemoryDataset):

    """
    A PyTorch InMemoryDataset to build multi-view dataset through graph data augmentation
    """

    def __init__(self, root="data", dataset='cora', transform=None, pre_transform=None):
        self.root, self.dataset, self.data_dir = download_data(root=root, dataset=dataset)
        create_dirs(self.dirs)
        super().__init__(root=self.data_dir, transform=transform, pre_transform=pre_transform)
        path = osp.join(self.processed_dir, self.processed_file_names[0])
        self.data, self.slices = torch.load(path)

    def process_full_batch_data(self, data):

        print("Processing full batch data")
        nodes = torch.tensor(np.arange(data.num_nodes), dtype=torch.long)

        data = Data(nodes=nodes, edge_index=data.edge_index, edge_attr=data.edge_attr, x=data.x, y=data.y,
                    train_mask1=data.train_mask1, val_mask1=data.val_mask1, test_mask1=data.test_mask1,
                    train_mask2=data.train_mask2, val_mask2=data.val_mask2, test_mask2=data.test_mask2,
                    train_mask3=data.train_mask3, val_mask3=data.val_mask3, test_mask3=data.test_mask3,
                    train_mask4=data.train_mask4, val_mask4=data.val_mask4, test_mask4=data.test_mask4,
                    train_mask5=data.train_mask5, val_mask5=data.val_mask5, test_mask5=data.test_mask5,
                    num_nodes=data.num_nodes)

        return [data]

    def process(self):
        """
        Process either a full batch or cluster data.

        :return:
        """
        processed_path = osp.join(self.processed_dir, self.processed_file_names[0])
        if not osp.exists(processed_path):
            path = osp.join(self.raw_dir, self.raw_file_names[0])
            data, _ = torch.load(path)
            data_list = self.process_full_batch_data(data)

            data, slices = self.collate(data_list)
            torch.save((data, slices), processed_path)

    @property
    def raw_file_names(self):
        return ["data.pt"]

    @property
    def processed_file_names(self):
        return [f'byg.data.pt']

    @property
    def raw_dir(self):
        return osp.join(self.data_dir, "raw")
        # return osp.join(self.data_dir, "geom_gcn/raw")

    @property
    def processed_dir(self):
        return osp.join(self.data_dir, "processed")
        # return osp.join(self.data_dir, "geom_gcn/processed")

    @property
    def dirs(self):
        return [self.raw_dir, self.processed_dir]

    def download(self):
        pass


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
    # if name == 'cora' or name == 'citeseer':
    #     mask1 = Random_Nonzero_Masking(data, labels, label_rate=1)
    #     mask2 = Random_Nonzero_Masking(data, labels, label_rate=2)
    #     mask3 = Random_Nonzero_Masking(data, labels, label_rate=3)
    #     mask4 = Random_Nonzero_Masking(data, labels, label_rate=4)
    #     mask5 = Random_Nonzero_Masking(data, labels, label_rate=5)
    #
    #     # mask0_5 = Random_Nonzero_Masking(data, labels, label_rate=0.5)
    #     # mask1 = Random_Nonzero_Masking(data, labels, label_rate=1)
    #     # mask2 = Random_Nonzero_Masking(data, labels, label_rate=2)
    #
    #     data.train_mask1, data.val_mask1, data.test_mask1 = mask1
    #     data.train_mask2, data.val_mask2, data.test_mask2 = mask2
    #     data.train_mask3, data.val_mask3, data.test_mask3 = mask3
    #     data.train_mask4, data.val_mask4, data.test_mask4 = mask4
    #     data.train_mask5, data.val_mask5, data.test_mask5 = mask5
    #
    # elif name == 'pubmed':
    #     mask0_03 = Random_Nonzero_Masking(data, labels, label_rate=0.03)
    #     mask0_06 = Random_Nonzero_Masking(data, labels, label_rate=0.06)
    #     mask0_1 = Random_Nonzero_Masking(data, labels, label_rate=0.1)
    #
    #     data.train_mask0_03, data.val_mask0_03, data.test_mask0_03 = mask0_03
    #     data.train_mask0_06, data.val_mask0_06, data.test_mask0_06 = mask0_06
    #     data.train_mask0_1, data.val_mask0_1, data.test_mask0_1 = mask0_1
    #
    # elif name == 'Computers' or name == 'Photo':
    #     mask0_15 = Random_Nonzero_Masking(data, labels, label_rate=0.15)
    #     mask0_2 = Random_Nonzero_Masking(data, labels, label_rate=0.2)
    #     mask0_25 = Random_Nonzero_Masking(data, labels, label_rate=0.25)
    #
    #     data.train_mask0_15, data.val_mask0_15, data.test_mask0_15 = mask0_15
    #     data.train_mask0_2, data.val_mask0_2, data.test_mask0_2 = mask0_2
    #     data.train_mask0_25, data.val_mask0_25, data.test_mask0_25 = mask0_25
    #
    # elif name == 'chameleon':
    #     mask0_5 = Random_Nonzero_Masking(data, labels, label_rate=0.5)
    #     mask1 = Random_Nonzero_Masking(data, labels, label_rate=1)
    #     mask2 = Random_Nonzero_Masking(data, labels, label_rate=2)
    #
    #     data.train_mask0_5, data.val_mask0_5, data.test_mask0_5 = mask0_5
    #     data.train_mask1, data.val_mask1, data.test_mask1 = mask1
    #     data.train_mask2, data.val_mask2, data.test_mask2 = mask2
    #
    # elif name == 'squirrel' or name == 'actor':
    #     mask1 = Random_Nonzero_Masking(data, labels, label_rate=1)
    #     mask2 = Random_Nonzero_Masking(data, labels, label_rate=2)
    #     mask4 = Random_Nonzero_Masking(data, labels, label_rate=4)
    #
    #     data.train_mask1, data.val_mask1, data.test_mask1 = mask1
    #     data.train_mask2, data.val_mask2, data.test_mask2 = mask2
    #     data.train_mask4, data.val_mask4, data.test_mask4 = mask4
                
    return data

def Random_Nonzero_Masking(data, labels, label_rate):
    # label_rate *= 0.01
    # train_size = int(data.x.size(0) * label_rate)
    train_size = np.unique(labels).shape[0] * label_rate
    eval_size = data.x.size(0) - train_size

    dev_size = int(eval_size * 0.1)

    final_train_mask = None;
    final_val_mask = None;
    final_test_mask = None
    cnt = 0
    while True:
        labels = data.y.numpy()
        # 0-2708随机排序  ### Cora
        perm = np.random.permutation(labels.shape[0])
        train_index = []
        for i in range(np.unique(labels).shape[0]):
            indices = np.where(labels == i)[0]
            if len(indices) >= label_rate:
                train_index.extend(np.random.choice(indices, size=label_rate, replace=False))
            else:
                train_index.extend(indices)
        data_index = np.arange(labels.shape[0])
        remaining_index = np.setdiff1d(data_index, train_index)
        dev_index = remaining_index[: dev_size]
        test_index = remaining_index[dev_size:]
        train_mask = torch.tensor(np.in1d(data_index, train_index), dtype=torch.bool)
        dev_mask = torch.tensor(np.in1d(data_index, dev_index), dtype=torch.bool)
        test_mask = torch.tensor(np.in1d(data_index, test_index), dtype=torch.bool)
        # train_index = perm[:train_size]
        # dev_index = perm[train_size: train_size + dev_size]
        # test_index = perm[train_size + dev_size:]
        # [0, 1, 2, ……, 2707]
        # data_index = np.arange(labels.shape[0])
        # # numpy.in1d(arr1, arr2, assume_unique = False, invert = False)
        # # 测试一个一维数组中的每个元素是否也存在于第二个数组中，并返回一个与arr1相同长度的布尔数组，当arr1的一个元素在arr2中时为真，
        # # 否则为假。
        # train_mask = torch.tensor(np.in1d(data_index, train_index), dtype=torch.bool)
        # dev_mask = torch.tensor(np.in1d(data_index, dev_index), dtype=torch.bool)
        # test_mask = torch.tensor(np.in1d(data_index, test_index), dtype=torch.bool)

        train_mask = train_mask.reshape(1, -1)
        test_mask = test_mask.reshape(1, -1)
        dev_mask = dev_mask.reshape(1, -1)
        # np.unique(labels).shape[0] = 7
        if np.unique(labels).shape[0] == np.unique(labels[train_mask[0]]).shape[0]:
            cnt += 1
        else:
            continue

        if final_train_mask is None:
            final_train_mask = train_mask
            final_val_mask = dev_mask
            final_test_mask = test_mask
        else:
            final_train_mask = torch.cat((final_train_mask, train_mask), dim=0)
            final_val_mask = torch.cat((final_val_mask, dev_mask), dim=0)
            final_test_mask = torch.cat((final_test_mask, test_mask), dim=0)

        if cnt == 20:
            break

    return final_train_mask, final_val_mask, final_test_mask

