import sys
import os

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# print(path_root)
sys.path.append(path_root)
from config import CUDA_VISIBLE_DEVICES, USE_TORCH, CPU_NUMS  # from config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3072"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TORCH"] = USE_TORCH
os.environ["VECLIB_MAXIMUM_THREADS"] = CPU_NUMS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = CPU_NUMS  # export OPENBLAS_NUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = CPU_NUMS  # export NUMEXPR_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = CPU_NUMS  # export MKL_NUM_THREADS=1
os.environ["OMP_NUM_THREADS"] = CPU_NUMS  # export OMP_NUM_THREADS=1

import torch
import torch.nn as nn

import torch.nn.functional as F


def gumbel_softmax(logits, tau=1.0, hard=False):
    # 从 Gumbel 分布中采样
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    y = F.softmax((logits + gumbel_noise) / tau, dim=-1)

    if hard:
        # 硬选择，将输出转化为 one-hot 向量
        y_hard = torch.zeros_like(y).scatter_(1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y  # 保持梯度
    return y

# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         # self.fc0 = nn.Linear(128256*sequence_length, 128256, bias=False)
#         self.fc1 = nn.Linear(128256, 4096, bias=False)
#         self.fc2 = nn.Linear(4096, 1024, bias=False)
#         self.fc3 = nn.Linear(1024, 256, bias=False)
#         self.fc4 = nn.Linear(256, 64, bias=False)
#         self.fc5 = nn.Linear(64, 16, bias=False)
#         self.fc6 = nn.Linear(16, 4, bias=False)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         # x = self.relu(self.fc0(x))
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.relu(self.fc3(x))
#         x = self.relu(self.fc4(x))
#         x = self.relu(self.fc5(x))
#         x = self.fc6(x)
#         return x
#
#
# class ModifiedModel(nn.Module):
#     def __init__(self, llm_model, mlp, sequence_length, output_dim):
#         super(ModifiedModel, self).__init__()
#         self.model = llm_model
#         self.mlp = mlp
#         self.fc_1 = nn.Linear(sequence_length*4, sequence_length, bias=False)
#         self.fc_2 = nn.Linear(sequence_length, 1024, bias=False)
#         self.fc_3 = nn.Linear(1024, 128, bias=False)
#         self.fc_4 = nn.Linear(128, output_dim, bias=False)
#         self.relu = nn.ReLU()
#
#     def forward(self, input_ids, attention_mask=None, labels=None):
#         # 获取 PeftModelForCausalLM 的输出
#         outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
#
#         # 将 logits 传入 MLP
#         logits = outputs.logits
#         # logits_1d = logits.view(logits.size(0), -1)# 假设 logits 是 (batch_size, seq_length, hidden_size)
#         mlp_output = self.mlp(logits)
#         out = self.relu(self.fc_1(mlp_output.view(mlp_output.size(0), -1)))
#         out = self.relu(self.fc_2(out))
#         out = self.relu(self.fc_3(out))
#         out = self.fc_4(out)
#         # 返回与原模型相同的格式，以便 Trainer 训练
#         # return mlp_output
#         return F.log_softmax(out, dim=-1), gumbel_softmax(out, tau=1.0, hard=True)


# tokenizer = LLMTokenizer.from_pretrained(PATH_MODEL_PRETRAIN_value, add_eos_token=True, trust_remote_code=True)
# ID_START = 128006
# ID_END = 128007
# ID_BOS = 128000
# ID_EOS = 128009
# ID_PAD = ID_EOS
# ID_BR = 198  # "\n"
# ID_SYSTEM = 9125
# ID_MODEL = 78191
# ID_USER = 882
# tokenizer.pad_token_id = ID_EOS
# tokenizer.eos_token_id = ID_EOS
# tokenizer.padding_side = "right"  # NO use attention-mask
# STOP_WORDS_IDS = [[ID_BOS], [ID_EOS], [ID_END]]
#
# model = LLMModel.from_pretrained(PATH_MODEL_PRETRAIN_value, torch_dtype=torch.bfloat16)
# model.is_parallelizable = IS_PARALLELIZABLE
# model.model_parallel = MODEL_PARALLEL
# model.config.use_cache = USE_CACHE
# config = LoraConfig(target_modules=TARGET_MODULES_policy,
#                     lora_dropout=LORA_DROPOUT,
#                     lora_alpha=LORA_ALPHA,
#                     task_type="CAUSAL_LM",  # 用来指定 LoRA 要适用于的任务类型。不同的任务类型会影响模型中的哪些部分应用 LoRA 以及如何配置 LoRA。根据不同的任务，LoRA 的配置方式可能会有所不同，特别是在模型的某些特定模块（如自注意力层）上。
#                     bias="none",
#                     r=LORA_R,
#                     )
# model = get_peft_model(model, config)
# input_text = {"role": "user",
#                   "content": "There is a graph consisting of webpages (nodes) and the hyperlinks (edges) between them. " +
#                              "There are four names of teacher networks: ['gprgnn', 'dirgnn', 'h2gcn', 'holognn']. "
#                              "What’s the best teacher network assignment result for the target webpage (node) based on the following information? "
#                              "You must output it in this format please: {'teacher': <your first answer>}, and don't need to give reasons and process for your reasoning. "}
# inputs = tokenizer(
#     input_text['content'],
#     return_tensors="pt",  # 返回 PyTorch tensor 形式，或者选择 'tf' 返回 TensorFlow tensor
#     padding="max_length",  # 对齐文本长度以适应模型输入的固定长度
#     truncation=True,       # 截断长于 max_length 的文本
#     max_length=1024          # 设定最大输入长度，例如 64 个 token
# )
# # out = model(**inputs)
# print(model)

# mlp = MLP(128256)
# merge_model = ModifiedModel(model, mlp, 1024)
# out = merge_model(**inputs)
# print(out.shape)
# print(out.shape)
# print(merge_model)
# print('#################################################################################')
# for name, param in merge_model.named_parameters():
#     if param.requires_grad:
#         print(name)
#     # print((name, param.data.dtype, param.requires_grad))
# print('#################################################################################')




class MLP_P(nn.Module):
    def __init__(self):
        super(MLP_P, self).__init__()
        # self.fc0 = nn.Linear(128256*sequence_length, 128256, bias=False)
        # self.fc1 = nn.Linear(128256, 4096, bias=False)
        self.fc1 = nn.Linear(1024, 256, bias=False)
        # self.fc1 = nn.Linear(1024, 1024, bias=False)
        # self.fc2 = nn.Linear(4096, 1024, bias=False)
        # self.fc3 = nn.Linear(4096, 256, bias=False)
        # self.fc3 = nn.Linear(1024, 256, bias=False)
        # self.fc4 = nn.Linear(256, 64, bias=False)
        # self.fc5 = nn.Linear(64, 16, bias=False)
        self.fc6 = nn.Linear(256, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        # x = self.relu(self.fc4(x))
        # x = self.relu(self.fc5(x))
        out = self.fc6(x)
        return F.softmax(out, dim=-1)

class MLP_V(nn.Module):
    def __init__(self):
        super(MLP_V, self).__init__()
        # self.fc0 = nn.Linear(128256*sequence_length, 128256, bias=False)
        # self.fc1 = nn.Linear(128256, 4096, bias=False)
        self.fc1 = nn.Linear(1024, 256, bias=False)
        # self.fc2 = nn.Linear(4096, 256, bias=False)
        # self.fc1 = nn.Linear(1024, 256, bias=False)
        # self.fc2 = nn.Linear(4096, 1024, bias=False)
        # self.fc3 = nn.Linear(1024, 256, bias=False)
        self.fc4 = nn.Linear(256, 1, bias=False)
        # self.fc5 = nn.Linear(64, 16, bias=False)
        # self.fc6 = nn.Linear(64, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        # x = self.relu(self.fc5(x))
        # out = self.fc6(x)
        return x


class Merge_Model(nn.Module):
    def __init__(self, llm_model, mlp_p, mlp_v, sequence_length):
        super(Merge_Model, self).__init__()
        self.model = llm_model
        self.mlp_p = mlp_p
        self.mlp_v = mlp_v

        # 冻结 LLM 模型的参数
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]  # 获取最后一个隐藏层
        cls_embedding = last_hidden_state[:, 0, :]  # 取 [CLS] token 的表示
        p_output = self.mlp_p(cls_embedding)  # 通过 MLP
        v_output = self.mlp_v(cls_embedding)

        return p_output, v_output


class MergeModel(nn.Module):
    def __init__(self, llm_model, mlp, sequence_length):
        super(MergeModel_P, self).__init__()
        self.model = llm_model
        self.mlp = mlp
        self.fc0 = nn.Linear(sequence_length*4, 256, bias=False)
        # self.fc1 = nn.Linear(256, 64, bias=False)
        self.fc2 = nn.Linear(256, 4, bias=False)
        self.relu = nn.ReLU()

        # 冻结 self.model 的参数
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        # 获取 PeftModelForCausalLM 的输出
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

        # 将 logits 传入 MLP
        logits = outputs.logits
        # logits_1d = logits.view(logits.size(0), -1)# 假设 logits 是 (batch_size, seq_length, hidden_size)
        mlp_output = self.mlp(logits)
        out = self.relu(self.fc0(mlp_output.view(mlp_output.size(0), -1)))
        # out = self.relu(self.fc1(out))
        # out = self.relu(self.fc_3(out))
        out = self.fc2(out)
        # 返回与原模型相同的格式，以便 Trainer 训练
        # return mlp_output
        return F.softmax(out, dim=-1)

class MergeModel_P(nn.Module):
    def __init__(self, llm_model, mlp, sequence_length):
        super(MergeModel_P, self).__init__()
        self.model = llm_model
        self.mlp = mlp
        self.fc0 = nn.Linear(sequence_length*4, 256, bias=False)
        # self.fc1 = nn.Linear(256, 64, bias=False)
        self.fc2 = nn.Linear(256, 4, bias=False)
        self.relu = nn.ReLU()

        # 冻结 self.model 的参数
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        # 获取 PeftModelForCausalLM 的输出
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

        # 将 logits 传入 MLP
        logits = outputs.logits
        # logits_1d = logits.view(logits.size(0), -1)# 假设 logits 是 (batch_size, seq_length, hidden_size)
        mlp_output = self.mlp(logits)
        out = self.relu(self.fc0(mlp_output.view(mlp_output.size(0), -1)))
        # out = self.relu(self.fc1(out))
        # out = self.relu(self.fc_3(out))
        out = self.fc2(out)
        # 返回与原模型相同的格式，以便 Trainer 训练
        # return mlp_output
        return F.softmax(out, dim=-1)

class Merge_Model_V(nn.Module):
    def __init__(self, llm_model, mlp, sequence_length):
        super(Merge_Model_V, self).__init__()
        self.model = llm_model
        self.mlp = mlp


        # 冻结 self.model 的参数
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]  # 获取最后一个隐藏层
        cls_embedding = last_hidden_state[:, 0, :]  # 取 [CLS] token 的表示
        mlp_output = self.mlp(cls_embedding)  # 通过 MLP
        # out = self.relu(self.fc0(mlp_output.view(mlp_output.size(0), -1)))
        # # out = self.relu(self.fc1(out))
        # # out = self.relu(self.fc_3(out))
        # out = self.fc2(out)
        # 返回与原模型相同的格式，以便 Trainer 训练
        # return mlp_output
        return mlp_output

class MergeModel_V(nn.Module):
    def __init__(self, llm_model, mlp, sequence_length):
        super(MergeModel_V, self).__init__()
        self.model = llm_model
        self.mlp = mlp
        self.fc0 = nn.Linear(sequence_length, 256, bias=False)
        # self.fc1 = nn.Linear(256, 64, bias=False)
        self.fc2 = nn.Linear(256, 1, bias=False)
        self.relu = nn.ReLU()

        # 冻结 self.model 的参数
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        # 获取 PeftModelForCausalLM 的输出
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

        # 将 logits 传入 MLP
        logits = outputs.logits
        # logits_1d = logits.view(logits.size(0), -1)# 假设 logits 是 (batch_size, seq_length, hidden_size)
        mlp_output = self.mlp(logits)
        out = self.relu(self.fc0(mlp_output.view(mlp_output.size(0), -1)))
        # out = self.relu(self.fc1(out))
        # out = self.relu(self.fc_3(out))
        out = self.fc2(out)
        # 返回与原模型相同的格式，以便 Trainer 训练
        # return mlp_output
        return out


class Model_P(nn.Module):
    def __init__(self):
        super(Model_P, self).__init__()
        # self.embedding = embeddings
        # self.fc1 = nn.Linear(128, 1024, bias=False)
        self.fc1 = nn.Linear(2048, 1024, bias=False)
        self.fc2 = nn.Linear(1024, 256, bias=False)
        self.fc3 = nn.Linear(256, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class Model_V(nn.Module):
    def __init__(self):
        super(Model_V, self).__init__()
        # self.embedding = embeddings
        # self.fc1 = nn.Linear(128, 1024, bias=False)
        self.fc1 = nn.Linear(2048, 1024, bias=False)
        self.fc2 = nn.Linear(1024, 256, bias=False)
        self.fc3 = nn.Linear(256, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x












