import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
import random
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
# 定义一个两层的二元分类器
class AttackModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(AttackModel, self).__init__()
        # 第一层：输入维度 -> 隐藏层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # 激活函数
        self.relu = nn.ReLU()
        # 第二层：隐藏层 -> 输出
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))  # 隐藏层和ReLU激活
        return torch.sigmoid(self.fc2(x))  # 输出层使用sigmoid
    
class GCNShadowModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNShadowModel, self).__init__()
        # 第一层GCN
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        # 第二层GCN
        self.gcn2 = GCNConv(hidden_dim, output_dim)
        # ReLU激活函数
        self.relu = nn.ReLU()
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index  # 获取节点特征和边信息
        
        # 第一层GCN卷积和激活
        x = self.relu(self.gcn1(x, edge_index))
        # 第二层GCN卷积，输出层
        x = self.gcn2(x, edge_index)
        
        return x
    
    
def train_shadow_model(model, data, learning_rate=0.005, epochs=100):
    # 使用交叉熵损失函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    criterion = nn.CrossEntropyLoss()
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练过程
    for epoch in range(epochs):
        model.train()
        
        # 将数据传入模型
        out = model(data)
        # 计算损失，data.y 是节点的真实标签
        loss = criterion(out[data.train_mask], data.y[data.train_mask])

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    model.eval()
    model = model.to('cpu')
    data = data.to('cpu')
    y_pred = model(data).detach().cpu()
    y = data.y
    y_pred = np.argmax(y_pred, axis=1)
    f1 = f1_score(y[data.test_mask], y_pred[data.test_mask], average="micro")
    
    print('F1 Score: {:.4f} '.format(f1))
    return model



def train_attack_model(train_x, input_dim, learning_rate=0.001, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建二元标签y，前一半为1，后一半为0
    num_samples = train_x.shape[0]
    half_samples = num_samples // 2
    y = torch.cat([torch.ones(half_samples), torch.zeros(num_samples - half_samples)]).to(device)

    # 初始化模型、损失函数和优化器
    model = AttackModel(input_dim)
    model = model.to(device)
    train_x = train_x.to(device)
    criterion = nn.BCELoss()  # 二元交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练过程
    for epoch in range(epochs):
        model.train()
        
        # 前向传播
        outputs = model(train_x).squeeze()  # 去掉多余维度
        loss = criterion(outputs, y)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:  # 每10个epoch打印一次损失
            predicted = (outputs > 0.5).float().cpu()  # 二元分类的阈值为0.5
            true_labels = y.cpu()

            # 计算F1 Score
            f1 = f1_score(true_labels, predicted)
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, F1: {f1:.4f}')
    
    print("Training finished.")
    return model

# 假设 train_x 已经是一个 Tensor，且其形状是 [num_samples, input_dim]
# 例如: train_attack_model(train_x, input_dim=train_x.shape[1])


def generate_shadow_model_output(shadow_model, data):
    # 设置模型为评估模式
    shadow_model.eval()
    
    # 获取训练集和测试集的节点索引
    train_indices = list(data.train_indices)
    test_indices = data.test_indices

    # 将test_indices_train按照0.8:0.2的比例分割
    num_test = len(test_indices)
    split_idx = int(num_test * 0.8)
    test_indices_train = test_indices[:split_idx]
    test_indices_val = test_indices[split_idx:]

    # 确保train_indices的大小与test_indices一致，随机下采样
    sampled_train_indices = random.sample(train_indices, len(test_indices_train))
    # 随机取出与test_indices_val长度相等的train_indices里的内容，并且要求与sampled_train_indices内容完全不同
    remaining_train_indices = list(set(train_indices) - set(sampled_train_indices))
    sampled_test_indices = random.sample(remaining_train_indices, len(test_indices_val))

    # 将train_indices和test_indices连接在一起
    combined_indices_train = torch.tensor(sampled_train_indices + test_indices_train, dtype=torch.long)
    combined_indices_test = torch.tensor(sampled_test_indices + test_indices_val, dtype=torch.long)

    # 获取这些节点的soft labels
    with torch.no_grad():
        soft_labels = F.softmax(shadow_model(data))  # 模型预测输出
        soft_labels_train = soft_labels[combined_indices_train]  # 选择对应节点的输出
        soft_labels_test = soft_labels[combined_indices_test]

    # soft_labels前一半是训练集节点，后一半是测试集节点
    print(f'Soft labels shape: {soft_labels_train.shape}')
    
    return soft_labels_train,soft_labels_test

def evaluate_attack_model(model, x):
    model.eval()  # 设置模型为评估模式
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_samples = x.shape[0]
    half_samples = num_samples // 2
    y = torch.cat([torch.ones(half_samples), torch.zeros(num_samples - half_samples)]).to(device)
    # 将数据迁移到对应的设备
    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        # 前向传播，获取模型的预测结果
        outputs = model(x).squeeze()  # 去掉多余的维度
    
    # 将输出转移到CPU，便于sklearn计算
    predicted_probs = outputs.cpu()
    true_labels = y.cpu()

    # 计算AUC
    auc_score = roc_auc_score(true_labels, predicted_probs)

    print(f"AUC Score: {auc_score:.4f}")
    return auc_score