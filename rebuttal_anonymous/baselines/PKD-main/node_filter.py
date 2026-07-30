import torch
import numpy as np
from get_paper_txt import get_text
from data_amazon import get_data
from src.data_wiki import get_data_wiki
from src.utils import masking
from read import get_raw_text_webkb
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def node_selecting(dataset_name, label_rate, ratio):
    dataset = dataset_name
    if dataset == 'cora':
        logit_1 = np.load('outputs/{}/{}/appnp_{}.npy'.format(dataset_name, label_rate, dataset_name))
        logit_2 = np.load('outputs/{}/{}/gcn_{}.npy'.format(dataset_name, label_rate, dataset_name))
        logit_3 = np.load('outputs/{}/{}/h2gcn_{}.npy'.format(dataset_name, label_rate, dataset_name))
        logit_4 = np.load('outputs/{}/{}/gcn2_{}.npy'.format(dataset_name, label_rate, dataset_name))
    elif dataset == 'pubmed':
        logit_1 = np.load('outputs/{}/{}/appnp_{}.npy'.format(dataset_name, label_rate, dataset_name))
        logit_2 = np.load('outputs/{}/{}/gcn_{}.npy'.format(dataset_name, label_rate, dataset_name))
        logit_3 = np.load('outputs/{}/{}/gat_{}.npy'.format(dataset_name, label_rate, dataset_name))
        logit_4 = np.load('outputs/{}/{}/h2gcn_{}.npy'.format(dataset_name, label_rate, dataset_name))
    elif dataset == 'wiki_cs':
        logit_1 = np.load('outputs/{}/{}/appnp_{}.npy'.format(dataset_name, label_rate, dataset_name))
        logit_2 = np.load('outputs/{}/{}/gcn_{}.npy'.format(dataset_name, label_rate, dataset_name))
        logit_3 = np.load('outputs/{}/{}/gat_{}.npy'.format(dataset_name, label_rate, dataset_name))
        logit_4 = np.load('outputs/{}/{}/gcn2_{}.npy'.format(dataset_name, label_rate, dataset_name))
    elif dataset == 'arxiv':
        logit_1 = np.array(torch.load('logits/semi/gcn_{}_best.pt'.format(2)))
        logit_2 = np.array(torch.load('logits/semi/gat_{}_best.pt'.format(2)))
        logit_3 = np.array(torch.load('logits/semi/appnp_{}_best.pt'.format(2)))
        logit_4 = np.array(torch.load('logits/semi/h2gcn_{}_best.pt'.format(2)))
    else:
        logit_1 = np.load('logit/{}/dirgnn_{}.npy'.format(label_rate, dataset))
        logit_2 = np.load('logit/{}/h2gcn_{}.npy'.format(label_rate, dataset))
        logit_3 = np.load('logit/{}/holognn_{}.npy'.format(label_rate, dataset))
        logit_4 = np.load('logit/{}/gprgnn_{}.npy'.format(label_rate, dataset))

    # 转换为概率分布
    prob1 = softmax(logit_1)
    prob2 = softmax(logit_2)
    prob3 = softmax(logit_3)
    prob4 = softmax(logit_4)

    # 防止log(0)，加上极小值
    epsilon = 1e-8
    prob1 += epsilon
    prob2 += epsilon
    prob3 += epsilon
    prob4 += epsilon

    # 计算每对之间的KL散度
    kl12 = np.sum(prob1 * np.log(prob1 / prob2), axis=1)
    # print(kl12)
    kl13 = np.sum(prob1 * np.log(prob1 / prob3), axis=1)
    # print(kl13)
    kl14 = np.sum(prob1 * np.log(prob1 / prob4), axis=1)
    # print(kl14)
    kl23 = np.sum(prob2 * np.log(prob2 / prob3), axis=1)
    # print(kl23)
    kl24 = np.sum(prob2 * np.log(prob2 / prob4), axis=1)
    # print(kl24)
    kl34 = np.sum(prob3 * np.log(prob3 / prob4), axis=1)
    # print(kl34)

    # 总的散度值
    total_kl = kl12 + kl13 + kl14 + kl23 + kl24 + kl34

    # 选择前a%的样本
    a = ratio  # 例如选择前10%的样本
    num_samples = int(0.01 * a * logit_1.shape[0])

    # 获取散度值从大到小的索引
    indices = np.argsort(total_kl)[::-1]
    top_indices = indices[:num_samples]

    # 生成mask数组
    mask = np.zeros(logit_1.shape[0], dtype=int)
    mask[top_indices] = 1
    one_tensor = torch.tensor(mask, dtype=torch.int32)
    mask_tensor = one_tensor.to(torch.bool)
    indices = torch.where(mask_tensor)
    index_list = indices[0].tolist()

    return mask_tensor, mask_tensor.to(torch.bool), index_list


def mask2indices(dataset_name, label_rate):
    if dataset_name == 'amazon':
        data = get_data()
        mask = masking(0, data, label_rate)[0]
        indices = torch.where(mask)
        index_list = indices[0].tolist()
    elif dataset_name == 'cora':
        data, _ = get_text(dataset_name)
        mask = masking(0, data, label_rate)[0]
        indices = torch.where(mask)
        index_list = indices[0].tolist()
    elif dataset_name == 'pubmed':
        data, _ = get_text(dataset_name)
        mask = masking(0, data, label_rate)[0]
        indices = torch.where(mask)
        index_list = indices[0].tolist()
    elif dataset_name == 'wiki_cs':
        data = get_data_wiki()
        mask = masking(0, data, label_rate)[0]
        indices = torch.where(mask)
        index_list = indices[0].tolist()
    elif dataset_name == 'arxiv':
        train_mask = torch.load('/share/home/u20526/wx/2p/mask/train_mask_2_{}.pt'.format(label_rate))
        index_list = torch.nonzero(train_mask).squeeze().tolist()
    else:
        data, _ = get_raw_text_webkb(dataset=dataset_name, use_text=True, seed=0)
        mask = data.train_mask
        indices = torch.where(mask)
        index_list = indices[0].tolist()
    return index_list



# label_rate = 5
# dataset_name = 'cora'
# # data = get_data()
# # data, _ = get_raw_text_webkb(dataset=dataset_name, use_text=True, seed=0)
# features = torch.tensor(np.load('{}_feature.npy'.format(dataset_name))).cuda()
# edge_index = torch.tensor(np.load('{}_edge_index.npy'.format(dataset_name))).cuda()
# labels_true = torch.tensor(np.load('{}_labels.npy'.format(dataset_name))).cuda()
# labels = torch.tensor(np.load('predicted_{}_labels_all.npy'.format(dataset_name))).cuda()
# true_index = mask2indices(dataset_name, label_rate)
# one_tensor, mask_tensor, index_list = node_selecting(dataset_name, label_rate, 60)
# print(true_index)
# print(index_list)
# for i in true_index:
#     mask_tensor[i] = True
#     labels[i] = labels_true[i]
#
# index_list = list(set(true_index + index_list))
# index_list.sort()
# print(index_list)
# j = 0
# for k in index_list:
#     if labels[k] == labels_true[k]:
#         j += 1
# print(j / len(index_list))



