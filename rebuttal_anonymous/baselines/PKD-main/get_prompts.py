from neighbor_select import get_indices_list
import torch
import numpy as np
from get_paper_txt import get_text

def prompt_collections(dataset, label_rate):
    dataset_name = dataset
    if dataset == 'amazon':
        logit_1 = np.load('logit/{}/dirgnn_{}.npy'.format(label_rate, dataset))
        logit_2 = np.load('logit/{}/h2gcn_{}.npy'.format(label_rate, dataset))
        logit_3 = np.load('logit/{}/holognn_{}.npy'.format(label_rate, dataset))
        logit_4 = np.load('logit/{}/gprgnn_{}.npy'.format(label_rate, dataset))
    elif dataset == 'cora':
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
    # neighbors_list = get_indices_list(logit_1, logit_2, logit_3, logit_4)
    logits = torch.tensor(np.stack([logit_1, logit_2, logit_3, logit_4], axis=0))
    #
    # if dataset_name == 'wiki_cs':
    #     with open('wikics_tokens.txt') as f:
    #         clean_texts = f.read().split('\n\n')
    # elif dataset_name == 'amazon':
    #     with open('amazon_5000_des.txt', 'r') as g:
    #         clean_texts = g.read().split('\n')
    # elif dataset_name == 'cora':
    #     data, clean_texts = get_text(dataset_name)
    # elif dataset_name == 'pubmed':
    #     data, clean_texts = get_text(dataset_name)
    # else:
    #     # with open('txt/{}_llama3-1.txt'.format(dataset), 'r') as f:
    #     #         clean_texts = f.readlines()
    #     with open('txt/{}_short.txt'.format(dataset), 'r') as k:
    #         clean_texts = k.readlines()
    #
    # messages = []
    #
    # for i in range(len(clean_texts)):
    #     message = []
    #     node_index = i
    #     text = clean_texts[node_index]
    #     role_1 = {"role": "system",
    #               "content": "You are a machine learning expert about teacher assignment for every target node in the process of graph knowledge distillation."}
    #     if dataset == 'cornell' or 'texas' or 'wisconsin' or 'washington':
    #         role_2 = {"role": "user",
    #                   "content": "There is a graph consisting of webpages (nodes) and the hyperlinks (edges) between them. " +
    #                              "There are four names of teacher networks: ['gprgnn', 'dirgnn', 'h2gcn', 'holognn']. "
    #                              "What’s the best teacher network assignment result for the target webpage (node) based on the following information? "
    #                              "You must output it in this format please: {'teacher': <your first answer>}, and don't need to give reasons and process for your reasoning. "
    #                              "Firstly, it is the content description of this target webpage: {" + text + "}."
    #                              "The dirgnn's logits output of this target webpage is {" + str(logit_1[node_index]) + "}."
    #                              "The gprgnn's logits output of this target webpage is {" + str(logit_4[node_index]) + "}. "
    #                              "The h2gcn's logits output of this target webpage is {" + str(logit_2[node_index]) + "}."
    #                              "The holognn's logits output of this target webpage is {" + str(logit_3[node_index]) + "}. "
    #                              "Then, it has following important neighbors (webpages), which are closely related the target webpage. Their content descriptions are: "}
    #
    #     elif dataset == 'amazon':
    #         role_2 = {"role": "user",
    #                   "content": "There is a graph consisting of products (nodes) and the co-purchase relationship (edges) between them. " +
    #                              "There are four names of teacher networks: ['gprgnn', 'dirgnn', 'h2gcn', 'holognn']. "
    #                              "What’s the best teacher network assignment result for the target product (node) based on the following information? "
    #                              "You must output it in this format please: {'teacher': <your first answer>}, and don't need to give reasons and process for your reasoning. "
    #                              "Firstly, it is the comments of this target product: {" + text + "}. "
    #                             "The dirgnn's logits output of this target product is {" + str(logit_1[node_index]) + "}."
    #                            "The gprgnn's logits output of this target product is {" + str(logit_4[node_index]) + "}. "
    #                            "The h2gcn's logits output of this target product is {" + str(logit_2[node_index]) + "}."
    #                            "The holognn's logits output of this target product is {" + str(logit_3[node_index]) + "}. "
    #                            "Then, it has following important neighbors (products), which are co-purchased with the target product. Their comments descriptions are: "}
    #     elif dataset == 'cora':
    #         role_2 = {"role": "user",
    #                   "content": "There is a graph consisting of papers (nodes) and the citation relationship (edges) between them. " +
    #                              "There are four names of teacher networks: ['appnp', 'gcn', 'h2gcn', 'gcn2']. "
    #                              "What’s the best teacher network assignment result for the target paper (node) based on the following information? "
    #                              "You must output it in this format please: {'teacher': <your first answer>}, and don't need to give reasons and process for your reasoning. "
    #                              "Firstly, it is the comments of this target paper: {" + text + "}. "
    #                              "The appnp's logits output of this target paper is {" + str(logit_1[node_index]) + "}."
    #                              "The gcn's logits output of this target paper is {" + str(logit_4[node_index]) + "}. "
    #                              "The h2gcn's logits output of this target paper is {" + str(logit_2[node_index]) + "}."
    #                              "The gcn2's logits output of this target paper is {" + str(logit_3[node_index]) + "}. "
    #                              "Then, it has following important neighbors (papers), which are cited by the target paper. Their comments descriptions are: "}
    #     elif dataset == 'pubmed':
    #         role_2 = {"role": "user",
    #                   "content": "There is a graph consisting of papers (nodes) and the citation relationship (edges) between them. " +
    #                              "There are four names of teacher networks: ['appnp', 'gcn', 'h2gcn', 'gat']. "
    #                              "What’s the best teacher network assignment result for the target paper (node) based on the following information? "
    #                              "You must output it in this format please: {'teacher': <your first answer>}, and don't need to give reasons and process for your reasoning. "
    #                              "Firstly, it is the comments of this target paper: {" + text + "}. "
    #                              "The appnp's logits output of this target paper is {" + str(logit_1[node_index]) + "}."
    #                              "The gcn's logits output of this target paper is {" + str(logit_4[node_index]) + "}. "
    #                              "The h2gcn's logits output of this target paper is {" + str(logit_2[node_index]) + "}."
    #                              "The gat's logits output of this target paper is {" + str(logit_3[node_index]) + "}. "
    #                              "Then, it has following important neighbors (papers), which are cited by the target paper. Their comments descriptions are: "}
    #     elif dataset == 'wiki_cs':
    #         role_2 = {"role": "user",
    #                   "content": "There is a graph consisting of webpages (nodes) and the hyperlinks (edges) between them. " +
    #                              "There are four names of teacher networks: ['appnp', 'gcn', 'gat', 'h2gcn']. "
    #                              "What’s the best teacher network assignment result for the target webpage (node) based on the following information? "
    #                              "You must output it in this format please: {'teacher': <your first answer>}, and don't need to give reasons and process for your reasoning. "
    #                              "Firstly, it is the content description of this target webpage: {" + text + "}."
    #                              "The appnp's logits output of this target webpage is {" + str(logit_1[node_index]) + "}."
    #                              "The gcn's logits output of this target webpage is {" + str(logit_4[node_index]) + "}. "
    #                              "The gat's logits output of this target webpage is {" + str(logit_2[node_index]) + "}."
    #                              "The h2gcn's logits output of this target webpage is {" + str(logit_3[node_index]) + "}. "
    #                              "Then, it has following important neighbors (webpages), which are closely related the target webpage. Their content descriptions are: "}
    #     neighbor_list = neighbors_list[node_index]
    #     for j in range(len(neighbor_list)):
    #         neighbor_index = neighbor_list[j]
    #         # print
    #         neighbor_text = clean_texts[neighbor_index]
    #         role_2['content'] += "{[Index: " + str(neighbor_index) + "]. Its description is [" + neighbor_text + "]}."
    #     messages.append(role_2['content'])
    # return messages, logits
    return logits

# me, lo = prompt_collections('amazon', 1)
# print(me)












# def prompt_collections(dataset, label_rate):
#     dataset_name = dataset
#     # if dataset == 'amazon':
#     #     logit_1 = np.load('logit/{}/dirgnn_{}.npy'.format(label_rate, dataset))
#     #     logit_2 = np.load('logit/{}/h2gcn_{}.npy'.format(label_rate, dataset))
#     #     logit_3 = np.load('logit/{}/holognn_{}.npy'.format(label_rate, dataset))
#     #     logit_4 = np.load('logit/{}/gprgnn_{}.npy'.format(label_rate, dataset))
#     # elif dataset == 'cora':
#     #     logit_1 = np.load('outputs/{}/{}/appnp_{}.npy'.format(dataset_name, label_rate, dataset_name))
#     #     logit_2 = np.load('outputs/{}/{}/gcn_{}.npy'.format(dataset_name, label_rate, dataset_name))
#     #     logit_3 = np.load('outputs/{}/{}/h2gcn_{}.npy'.format(dataset_name, label_rate, dataset_name))
#     #     logit_4 = np.load('outputs/{}/{}/gcn2_{}.npy'.format(dataset_name, label_rate, dataset_name))
#     # elif dataset == 'pubmed':
#     #     logit_1 = np.load('outputs/{}/{}/appnp_{}.npy'.format(dataset_name, label_rate, dataset_name))
#     #     logit_2 = np.load('outputs/{}/{}/gcn_{}.npy'.format(dataset_name, label_rate, dataset_name))
#     #     logit_3 = np.load('outputs/{}/{}/gat_{}.npy'.format(dataset_name, label_rate, dataset_name))
#     #     logit_4 = np.load('outputs/{}/{}/h2gcn_{}.npy'.format(dataset_name, label_rate, dataset_name))
#     # elif dataset == 'wiki_cs':
#     #     logit_1 = np.load('outputs/{}/{}/appnp_{}.npy'.format(dataset_name, label_rate, dataset_name))
#     #     logit_2 = np.load('outputs/{}/{}/gcn_{}.npy'.format(dataset_name, label_rate, dataset_name))
#     #     logit_3 = np.load('outputs/{}/{}/gat_{}.npy'.format(dataset_name, label_rate, dataset_name))
#     #     logit_4 = np.load('outputs/{}/{}/gcn2_{}.npy'.format(dataset_name, label_rate, dataset_name))
#     # else:
#     #     logit_1 = np.load('logit/{}/dirgnn_{}.npy'.format(label_rate, dataset))
#     #     logit_2 = np.load('logit/{}/h2gcn_{}.npy'.format(label_rate, dataset))
#     #     logit_3 = np.load('logit/{}/holognn_{}.npy'.format(label_rate, dataset))
#     #     logit_4 = np.load('logit/{}/gprgnn_{}.npy'.format(label_rate, dataset))
#     # neighbors_list = get_indices_list(logit_1, logit_2, logit_3, logit_4)
#     # logits = torch.tensor(np.stack([logit_1, logit_2, logit_3, logit_4], axis=0))
#     # with open('amazon_5000_des.txt', 'r') as g:
#     #     amazon_short_texts = g.read().split('\n')
#     if dataset_name == 'wiki_cs':
#         with open('wikics_tokens.txt') as f:
#             clean_texts = f.read().split('\n\n')
#     elif dataset_name == 'amazon':
#         with open('amazon_5000_des.txt', 'r') as g:
#             clean_texts = g.read().split('\n')
#     elif dataset_name == 'cora':
#         data, clean_texts = get_text(dataset_name)
#     elif dataset_name == 'pubmed':
#         data, clean_texts = get_text(dataset_name)
#     else:
#         # with open('txt/{}_llama3-1.txt'.format(dataset), 'r') as f:
#         #         clean_texts = f.readlines()
#         with open('txt/{}_short.txt'.format(dataset), 'r') as k:
#             clean_texts = k.readlines()
#
#     # messages = []
#     #
#     # for i in range(len(clean_texts)):
#     #     message = []
#     #     node_index = i
#     #     text = clean_texts[node_index]
#     #     role_1 = {"role": "system",
#     #               "content": "You are a machine learning expert about teacher assignment for every target node in the process of graph knowledge distillation."}
#     #     if dataset == 'cornell' or 'texas' or 'wisconsin' or 'washington':
#     #         role_2 = {"role": "user",
#     #                   "content": "There is a graph consisting of webpages (nodes) and the hyperlinks (edges) between them. " +
#     #                              "There are four names of teacher networks: ['gprgnn', 'dirgnn', 'h2gcn', 'holognn']. "
#     #                              "What’s the best teacher network assignment result for the target webpage (node) based on the following information? "
#     #                              "You must output it in this format please: {'teacher': <your first answer>}, and don't need to give reasons and process for your reasoning. "
#     #                              "Firstly, it is the content description of this target webpage: {" + text + "}."
#     #                              "The dirgnn's logits output of this target webpage is {" + str(logit_1[node_index]) + "}."
#     #                              "The gprgnn's logits output of this target webpage is {" + str(logit_4[node_index]) + "}. "
#     #                              "The h2gcn's logits output of this target webpage is {" + str(logit_2[node_index]) + "}."
#     #                              "The holognn's logits output of this target webpage is {" + str(logit_3[node_index]) + "}. "
#     #                              "Then, it has following important neighbors (webpages), which are closely related the target webpage. Their content descriptions are: "}
#     #
#     #     elif dataset == 'amazon':
#     #         role_2 = {"role": "user",
#     #                   "content": "There is a graph consisting of products (nodes) and the co-purchase relationship (edges) between them. " +
#     #                              "There are four names of teacher networks: ['gprgnn', 'dirgnn', 'h2gcn', 'holognn']. "
#     #                              "What’s the best teacher network assignment result for the target product (node) based on the following information? "
#     #                              "You must output it in this format please: {'teacher': <your first answer>}, and don't need to give reasons and process for your reasoning. "
#     #                              "Firstly, it is the comments of this target product: {" + text + "}. "
#     #                             "The dirgnn's logits output of this target product is {" + str(logit_1[node_index]) + "}."
#     #                            "The gprgnn's logits output of this target product is {" + str(logit_4[node_index]) + "}. "
#     #                            "The h2gcn's logits output of this target product is {" + str(logit_2[node_index]) + "}."
#     #                            "The holognn's logits output of this target product is {" + str(logit_3[node_index]) + "}. "
#     #                            "Then, it has following important neighbors (products), which are co-purchased with the target product. Their comments descriptions are: "}
#     #     elif dataset == 'cora':
#     #         role_2 = {"role": "user",
#     #                   "content": "There is a graph consisting of papers (nodes) and the citation relationship (edges) between them. " +
#     #                              "There are four names of teacher networks: ['appnp', 'gcn', 'h2gcn', 'gcn2']. "
#     #                              "What’s the best teacher network assignment result for the target paper (node) based on the following information? "
#     #                              "You must output it in this format please: {'teacher': <your first answer>}, and don't need to give reasons and process for your reasoning. "
#     #                              "Firstly, it is the comments of this target paper: {" + text + "}. "
#     #                              "The appnp's logits output of this target paper is {" + str(logit_1[node_index]) + "}."
#     #                              "The gcn's logits output of this target paper is {" + str(logit_4[node_index]) + "}. "
#     #                              "The h2gcn's logits output of this target paper is {" + str(logit_2[node_index]) + "}."
#     #                              "The gcn2's logits output of this target paper is {" + str(logit_3[node_index]) + "}. "
#     #                              "Then, it has following important neighbors (papers), which are cited by the target paper. Their comments descriptions are: "}
#     #     elif dataset == 'pubmed':
#     #         role_2 = {"role": "user",
#     #                   "content": "There is a graph consisting of papers (nodes) and the citation relationship (edges) between them. " +
#     #                              "There are four names of teacher networks: ['appnp', 'gcn', 'h2gcn', 'gat']. "
#     #                              "What’s the best teacher network assignment result for the target paper (node) based on the following information? "
#     #                              "You must output it in this format please: {'teacher': <your first answer>}, and don't need to give reasons and process for your reasoning. "
#     #                              "Firstly, it is the comments of this target paper: {" + text + "}. "
#     #                              "The appnp's logits output of this target paper is {" + str(logit_1[node_index]) + "}."
#     #                              "The gcn's logits output of this target paper is {" + str(logit_4[node_index]) + "}. "
#     #                              "The h2gcn's logits output of this target paper is {" + str(logit_2[node_index]) + "}."
#     #                              "The gat's logits output of this target paper is {" + str(logit_3[node_index]) + "}. "
#     #                              "Then, it has following important neighbors (papers), which are cited by the target paper. Their comments descriptions are: "}
#     #     elif dataset == 'wiki_cs':
#     #         role_2 = {"role": "user",
#     #                   "content": "There is a graph consisting of webpages (nodes) and the hyperlinks (edges) between them. " +
#     #                              "There are four names of teacher networks: ['appnp', 'gcn', 'gat', 'h2gcn']. "
#     #                              "What’s the best teacher network assignment result for the target webpage (node) based on the following information? "
#     #                              "You must output it in this format please: {'teacher': <your first answer>}, and don't need to give reasons and process for your reasoning. "
#     #                              "Firstly, it is the content description of this target webpage: {" + text + "}."
#     #                              "The appnp's logits output of this target webpage is {" + str(logit_1[node_index]) + "}."
#     #                              "The gcn's logits output of this target webpage is {" + str(logit_4[node_index]) + "}. "
#     #                              "The gat's logits output of this target webpage is {" + str(logit_2[node_index]) + "}."
#     #                              "The h2gcn's logits output of this target webpage is {" + str(logit_3[node_index]) + "}. "
#     #                              "Then, it has following important neighbors (webpages), which are closely related the target webpage. Their content descriptions are: "}
#     #     neighbor_list = neighbors_list[node_index]
#     #     for j in range(len(neighbor_list)):
#     #         neighbor_index = neighbor_list[j]
#     #         neighbor_text = clean_texts[neighbor_index]
#     #         role_2['content'] += "{[Index: " + str(neighbor_index) + "]. Its description is [" + neighbor_text + "]}."
#     #     messages.append(role_2['content'])
#     return clean_texts
















