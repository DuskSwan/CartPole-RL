import torch
import torch.nn as nn
import torch.optim as optim

class QLinear(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()
        # 定义网络层
        self.fc1 = nn.Linear(observation_dim, 128) # 输入层到第一隐藏层
        self.relu1 = nn.ReLU() # 激活函数
        self.fc2 = nn.Linear(128, 128) # 第一隐藏层到第二隐藏层
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, action_dim) # 输出层，每个动作一个输出 Q 值

    def forward(self, state):
        # 定义前向传播
        x = self.relu1(self.fc1(state))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 定义主路径的线性层
        self.fc = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU() # 用于主路径的激活
        self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        # nn.Identity() 是一个不做任何操作的层，当维度相同时，直接使用 x

    def forward(self, x):
        # 保存原始输入，用于残差连接
        residual = self.shortcut(x)
        out = self.fc(x)
        out += residual 
        out = self.relu(out) # 放在这里更符合标准的 ResNet Block 结构

        return out

class QResNet(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dims=(128, 64, 32)):
        super().__init__()
        self.fc0 = nn.Linear(observation_dim, hidden_dims[0])
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims) - 1)
        ])
        self.fcn = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, state):
        x = self.relu(self.fc0(state))
        for block in self.blocks:
            x = block(x)
        return self.fcn(x)  # 输出每个动作的 Q 值