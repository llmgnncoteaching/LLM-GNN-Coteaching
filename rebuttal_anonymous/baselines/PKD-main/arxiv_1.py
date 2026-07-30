from model_merge import Model_P, Model_V
import torch
import torch.optim as optim
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
from models.gcn import GCN
import numpy as np
import random
from src.get_data_7 import get_data_tag
from torch.distributions import Categorical
import torch.nn.functional as F
from get_prompts import prompt_collections
from torch.cuda.amp import autocast
import gc
# from tete_p import eidx_to_sp
from args import parse_args
from node_filter import node_selecting, mask2indices
import time

def reset_parameters(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def eidx_to_sp(n: int, edge_index: torch.Tensor, device=None) -> torch.sparse.Tensor:
    indices = edge_index
    values = torch.FloatTensor([1.0] * len(edge_index[0])).to(edge_index.device)
    coo = torch.sparse_coo_tensor(indices=indices, values=values, size=[n, n])
    if device is None:
        device = edge_index.device
    return coo.to(device)

def kl_divergence(p, q):
    p = p + 1e-7
    q = q + 1e-7

    # 计算 KL 散度
    kl_loss = torch.sum(p * torch.log(p / q), dim=-1).mean()
    return kl_loss

def compute_accuracy_teacher(prediction, label):
    # _, prediction = student_model(feature, edge_index)
    correct = (prediction == label).sum().item()
    accuracy = correct / label.size(0) * 100
    return accuracy

def compute_accuracy_teacher_mask(prediction, label, index_list):
    correct = 0
    for index in index_list:
        if prediction[index] == label[index]:
            correct += 1
    accuracy = correct / len(index_list) * 100
    return accuracy

def distill(node_index, logits, student, features, adj_t, teacher_assignment, labels, mask, index_list, optimizer_s, e, num_epochs=3):
    teacher_logits = logits[:, node_index, :]
    selected_logit = teacher_assignment @ teacher_logits

    teacher_prob = torch.nn.functional.softmax(selected_logit, dim=-1)
    # reset_parameters(student)
    for epoch in range(num_epochs):
        student.train()
        optimizer_s.zero_grad()  # 每次迭代前清空梯度

        # student_logits, _, _ = student(eidx_to_sp(len(features), edge_index.detach().cpu()).to(device), features)
        student_logits, preds, _ = student_model(features, adj_t)
        # print(student_logits)
        student_logits_selected = student_logits[node_index].float()  # 确保是 float32

        student_prob = torch.nn.functional.softmax(student_logits_selected, dim=-1)
        # student_prob = student_logits_selected
        # ce_loss = F.nll_loss(student_logits[mask], labels[mask])
        # ce_loss = F.nll_loss(student_logits, labels)
        # ce_loss = F.nll_loss(student_prob, labels_true[node_index])
        ce_loss = F.nll_loss(student_prob, labels[node_index])
        # 计算 distill_loss 和 ent_loss
        distill_loss = kl_divergence(teacher_prob, student_prob)
        # ent_loss = (-student_prob * torch.log(student_prob + 1e-15)).sum(dim=-1).mean()

        # 计算总损失并反向传播
        # total_loss = distill_loss + ent_loss + ce_loss
        total_loss = distill_loss + ce_loss
        total_loss.backward()  # 不使用 retain_graph=True

        # 更新学生模型参数
        optimizer_s.step()
        # if (epoch+1) % 5 == 0:
        # print("Node Index: [{}/{}] Epoch: [{}/{}/{}] Student Model training: distill loss: {:.4f}, ent loss: {:.4f}, ce loss: {:.4f}, total loss: {:.4f}".format(node_index, len(index_list), e, 600, epoch+1, distill_loss.item(),
        #                                                                               ent_loss.item(), ce_loss.item(), total_loss.item()))
        print("Node Index: [{}/{}] Epoch: [{}/{}/{}] Student Model training: distill loss: {:.4f}, ce loss: {:.4f}, total loss: {:.4f}".format(node_index, len(index_list), e, 200, epoch+1, distill_loss.item(),
                                                                                      ce_loss.item(), total_loss.item()))
        torch.save(student.state_dict(), '{}_{}_{}_without_best_last_student.pt'.format(args.dataset_name, args.label_rate, ratio))
    # 在 no_grad 模式下评估学生模型
    # with torch.no_grad():
    # student_logits, pred, _ = student(eidx_to_sp(len(features), edge_index.detach().cpu()).to(device), features)
    student_logits, pred, _ = student_model(features, adj_t)
    # student_logits_selected = student_logits[node_index].float()
    # student_prob = torch.nn.functional.softmax(student_logits_selected, dim=-1)
    # kl_diff = kl_divergence(teacher_prob, student_prob)
    # ce_diff = F.nll_loss(student_logits[mask], labels[mask])
    acc = compute_accuracy_teacher_mask(pred, labels, index_list)
    accu = compute_accuracy_teacher(pred, labels_true)
    # diff = acc - ce_diff - kl_diff
    print(acc, accu)
    return acc

class PPOAgent:
    def __init__(self, model_policy, model_value, optimizer_1, optimizer_2, gamma=0.99, clip_epsilon=0.2):
        self.policy_net = model_policy
        self.value_net = model_value
        self.optimizer_policy = optimizer_1
        self.optimizer_value = optimizer_2
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

    def take_action(self, state):
        action_probs = self.policy_net(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_advantage(self, rewards, values):
        # 计算优势函数
        advantages = []
        cumulative_return = 0
        for reward, value in zip(reversed(rewards), reversed(values)):
            cumulative_return = reward + self.gamma * cumulative_return
            advantage = cumulative_return - value
            advantages.insert(0, advantage)
        return torch.tensor(advantages).cuda()

    def update(self, trajectories):
        # torch.autograd.set_detect_anomaly(True)
        # scaler_2 = GradScaler()
        for trajectory in trajectories:
            states_policy, states_value, actions, log_probs, rewards, values, epoch = trajectory
            rewards_r = torch.tensor(rewards, dtype=torch.float32, requires_grad=True).cuda()

            # 计算优势函数
            advantages = rewards - values
            advantages_r = torch.tensor(advantages, dtype=torch.float32, requires_grad=True).cuda()

            # 更新策略网络
            with autocast():
                probs = self.policy_net(states_policy)
                # probs = F.softmax(logits, dim=-1)
                distribution = Categorical(probs)
                new_log_probs = distribution.log_prob(torch.tensor(action).to(device))
                # actions, log_probs = self.policy_net(**states_policy)  # 重新计算 log_probs
                ratios = torch.exp(new_log_probs - log_probs.detach())
                print(ratios)
                surr1 = ratios * advantages_r
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_r
                policy_loss = -torch.min(surr1, surr2).mean()

                # 更新价值网络
                predicted_values = self.value_net(states_value)
                value_loss = F.mse_loss(predicted_values, rewards_r)

            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

            print("Epoch: {} Policy Loss: {} Value Loss: {} Value: {} Reward: {}".format(epoch, policy_loss.item(),
                                                                                         value_loss.item(),
                                                                                         predicted_values.item(),
                                                                                         rewards_r))
args = parse_args()
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
label_rate = 5
ratio = 60
dataset_name = 'arxiv'
# data = get_data()
dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())

data = dataset[0].to(device)
# data = get_data_tag(dataset_name)
# adj_t = data.adj_t
features = torch.tensor(np.load('{}_feature.npy'.format(dataset_name))).cuda()
edge_index = torch.tensor(np.load('{}_edge_index.npy'.format(dataset_name))).cuda()
labels_true = torch.tensor(np.load('{}_labels.npy'.format(dataset_name))).cuda()
labels = torch.tensor(np.load('predicted_{}_labels_all.npy'.format(dataset_name))).cuda()
# labels = torch.tensor(np.load('{}_without_finetune_labels.npy'.format(dataset_name))).cuda()
true_index = mask2indices(dataset_name, label_rate)
one_tensor, mask_tensor, index_list = node_selecting(dataset_name, label_rate, ratio)
for i in true_index:
    mask_tensor[i] = True
    labels[i] = labels_true[i]

index_list = list(set(true_index + index_list))
index_list.sort()

Policy_model = Model_P().to(device)
Value_model = Model_V().to(device)

# student_model = H2GCN(feat_dim=features.size(1), hidden_dim=128, class_dim=5).to(device)
student_model = GCN(features.size(1), 128, 40, 2, 0.5).to(device)
# student_model.load_state_dict(torch.load('{}_student_best_model_{}_h2gcn.pt'.format(dataset_name, label_rate), map_location=device))
student_model = student_model.to(device)
with open('{}_pseudo_without_{}_{}.txt'.format(dataset_name, label_rate, ratio), 'a') as f:
    with torch.no_grad():
        # student_logits, preds, _ = student_model(eidx_to_sp(len(features), edge_index.detach().cpu()).to(device), features)
        student_logits, preds, _ = student_model(features, data.adj_t)
        acc = compute_accuracy_teacher(preds, labels_true)
        f.write("First acc:")
        f.write('\n')
        f.write(str(acc))
        f.write('\n')
        f.close()

optimizer_s = optim.Adam(student_model.parameters(), lr=0.0001, weight_decay=1e-5)
optimizer_p = optim.Adam(Policy_model.parameters(), lr=0.0002, weight_decay=1e-4)
optimizer_v = optim.Adam(Value_model.parameters(), lr=0.0002, weight_decay=1e-4)

# prompts, logits = prompt_collections(dataset_name, label_rate)
logits = prompt_collections(dataset_name, label_rate)
logits = logits.to(device)
# print(logits.size())
# cls_tokens = np.load('{}_qwen_pca_{}.npy'.format(dataset_name, label_rate))
cls_tokens = np.load('/share/home/u20526/wx/walk_of_thoughts/arxiv_pca_1.npy')

# 将numpy数组转换为float32类型
cls_tokens = cls_tokens.astype(np.float32)

# 将numpy数组转换为torch tensor并指定为bfloat16类型
cls_tokens = torch.from_numpy(cls_tokens).to(device)

# 将tensor移动到CUDA设备上
embeddings = cls_tokens.cuda()
# print(embeddings.shape[0])
agent = PPOAgent(Policy_model, Value_model, optimizer_p, optimizer_v)
start_time = time.time()
# 获取GPU信息
gpu_name = torch.cuda.get_device_name(device)
gpu_memory_allocated = torch.cuda.memory_allocated(device)  # 当前GPU的内存使用量（字节）
gpu_memory_cached = torch.cuda.memory_cached(device)  # 当前GPU的内存缓存量（字节）
for epoch in range(3):
    trajectories = []
    for i in index_list:
        with autocast():
            value = agent.value_net(embeddings[i])
            action, log_prob = agent.take_action(embeddings[i])
            assignment = F.one_hot(torch.tensor(action), num_classes=4).float().to(device)
            print(assignment)
        reward = distill(i, logits, student_model, features, data.adj_t, assignment, labels, mask_tensor, index_list, optimizer_s, epoch)
        trajectories.append((embeddings[i], embeddings[i], action, log_prob, reward, value, epoch))
        end_time = time.time()

        # 计算总挂钟时间
        elapsed_time = end_time - start_time  # 总挂钟时间（秒）
        peak_memory = torch.cuda.max_memory_allocated(device)  # 峰值内存（字节）
        print(f"GPU Memory Allocated: {gpu_memory_allocated / 1024 ** 2:.2f} MB")
        print(f"GPU Memory Cached: {gpu_memory_cached / 1024 ** 2:.2f} MB")
        print(f"Peak Memory Usage: {peak_memory / 1024 ** 2:.2f} MB")
        print(f"Total Elapsed Time: {elapsed_time:.2f} seconds")

    agent.update(trajectories)
    student_model.load_state_dict(torch.load('{}_{}_{}_without_best_last_student.pt'.format(args.dataset_name, args.label_rate, ratio), map_location=device))
    with open('{}_pseudo_without_{}_{}.txt'.format(dataset_name, label_rate, ratio), 'a') as f:
        with torch.no_grad():
            # student_logits, preds, _ = student_model(eidx_to_sp(len(features), edge_index.detach().cpu()).to(device), features)
            student_logits, preds, _ = student_model(features, data.adj_t)
            acc = compute_accuracy_teacher(preds, labels_true)
            f.write(str(acc))
            f.write('\n')
            f.close()

end_time = time.time()

# 计算总挂钟时间
elapsed_time = end_time - start_time  # 总挂钟时间（秒）

# 获取峰值内存
peak_memory = torch.cuda.max_memory_allocated(device)  # 峰值内存（字节）

# 输出结果
print(f"GPU Name: {gpu_name}")

print(f"Total Elapsed Time: {elapsed_time:.2f} seconds")

# 重置CUDA的内存跟踪
torch.cuda.reset_max_memory_allocated(device)

