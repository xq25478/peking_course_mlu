# 第一课作业使用TensorLayerX训练MNIST数据集的MLP模型
from __future__ import print_function
import numpy as np
import torch
import torch_mlu
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../")))
from common.utils import load_imdb_dataset, Accuracy

from torch.nn import Linear
from torchvision.transforms import ToTensor

# 导入数据处理相关库
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# 构建神经网络模型

class CustomModel(torch.nn.Module):
    '''
    这是一个自定义的MLP模型
    '''

    def __init__(self):
        '''
        这个网络包含三层全连接层
        第一层是输入层, 输入数据的维度是784, 输出数据的维度是800
        第二层是隐藏层, 输入数据的维度是800, 输出数据的维度是800
        第三层是输出层, 输入数据的维度是800, 输出数据的维度是10, 代表10个类别
        '''
        super(CustomModel, self).__init__()  # 调用父类的构造函数

        # 使用Linear层构建全连接层
        self.linear1 = Linear(out_features=800, in_features=784)
        self.act1 = torch.nn.ReLU()
        self.linear2 = Linear(out_features=800, in_features=800)
        self.act2 = torch.nn.ReLU()
        self.linear3 = Linear(out_features=10, in_features=800)

    def forward(self, x):
        '''
        定义网络的前向传播过程
        '''
        x = torch.reshape(x, [-1, 784])
        a = self.linear1(x)
        a = self.act1(a)
        a = self.linear2(a)
        a = self.act2(a)
        out = self.linear3(a)

        return out


# 实例化模型
MLP = CustomModel()


metric = Accuracy()

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    log_interval = 16
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    train_acc = 0
    train_loss = 0
    n_iter= 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        metric.update(output,target)
        train_acc += metric.result()
        train_loss += loss.item()
        metric.reset()
        n_iter += 1 
    # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
    #     epoch, batch_idx * len(data), len(train_loader.dataset),
    #     100. * batch_idx / len(train_loader), loss.item(),train_acc / n_iter))
    print('Train Epoch: {} Loss: {:.6f}\tAcc: {:.6f}'.format(epoch, train_loss / n_iter ,train_acc / n_iter))


# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     # loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             # sum up batch loss
#             # test_loss += F.nll_loss(output, target, reduction='sum').item()
#             # test_loss += loss_func(output, target)
#             # get the index of the max log-probability
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     # test_loss /= len(test_loader.dataset)

#     print('\nTest set Accuracy: {}/{} ({:.0f}%)\n'.format( correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def main():
    use_mlu = torch.mlu.is_available()
    # 定义训练参数
    n_epoch = 50  # 训练50个epoch
    batch_size = 128  # 每个batch包含128个样本
    print_freq = 1  # 每训练1个epoch打印一次训练信息
    torch.manual_seed(0)

    device = torch.device("mlu:0")

    test_batch_size = 16
    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = MLP.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    gamma = 0.7
    # scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, n_epoch + 1):
        train(model, device, train_loader, optimizer, epoch)
        # test(model, device, test_loader)
        # scheduler.step()
    save_model = False
    if save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
