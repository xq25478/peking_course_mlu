# 第一课自定义激活函数的作业代码
import os
import torch
import torch_mlu


class ActSigmoid(torch.nn.Module):
    '''
    Sigmoid 激活函数
    '''

    def forward(self, x):
        '''
        前向传播, 数学公式：
        y = 1 / (1 + exp(-x))
        '''
        return 1 / (1 + torch.exp(-x))


class ActTanh(torch.nn.Module):
    '''
    Tanh 激活函数
    '''

    def forward(self, x):
        '''
        前向传播, 数学公式：
        y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        '''
        return torch.tanh(x)  # 直接调用torch.tanh()函数


class ActLeakyReLU(torch.nn.Module):
    '''
    LeakyReLU 激活函数
    '''

    def forward(self, x):
        '''
        前向传播, 数学公式：
        y = max(0.01*x, x)
        '''
        return torch.maximum(x, 0.01*x)


if __name__ == "__main__":
    # Test
    # 定义输入向量
    # x = torch.convert_to_tensor([[1., 2., 3.]])
    # w = torch.convert_to_tensor([[-0.5], [0.2], [0.1]])
    # b1 = torch.convert_to_tensor(0.5)

    device = torch.device("mlu:0" if torch.mlu.is_available() else "cpu")
    x = torch.FloatTensor([[1., 2., 3.]]).to(device)
    w = torch.FloatTensor([[-0.5], [0.2], [0.1]]).to(device)
    b1 = torch.FloatTensor([0.5]).to(device)

    # 矩阵乘法
    z1 = torch.matmul(x, w)+b1
    print("Z:\n", z1, "\nShape", z1.shape)

    # 测试激活函数
    # Sigmoid function
    sigmoid_torch = torch.nn.Sigmoid()  # torch内置的Sigmoid激活函数对象
    a1 = sigmoid_torch(z1)  # 调用对象的__call__方法，执行前向传播
    print("Result torch sigmoid:", a1)  # 使用torch内置函数的结果

    sigmoid_act = ActSigmoid() # 自定义的Sigmoid激活函数对象
    a2 = sigmoid_act(z1) # 调用对象的__call__方法，执行前向传播
    print("Result act sigmoid:", a2) # 使用自定义的激活函数的结果

    # Tanh
    tanh_torch = torch.nn.Tanh() # torch内置的Tanh激活函数对象
    a3 = tanh_torch(z1)  # 调用对象的__call__方法，执行前向传播
    print("Result torch Tanh:", a3) # 使用torch内置函数的结果

    tanh_act = ActTanh() # 自定义的Tanh激活函数对象
    a4 = tanh_act(z1) # 调用对象的__call__方法，执行前向传播
    print("Result act Tanh:", a4) # 使用自定义的激活函数的结果

    # Leaky ReLU
    leakyrelu_torch = torch.nn.LeakyReLU() # torch内置的LeakyReLU激活函数对象
    a5 = leakyrelu_torch(z1) # 调用对象的__call__方法，执行前向传播
    print("Result torch LeakyReLU:", a5) # 使用torch内置函数的结果

    leakyrelu_act = ActLeakyReLU() # 自定义的LeakyReLU激活函数对象
    a6 = leakyrelu_act(z1)  # 调用对象的__call__方法，执行前向传播
    print("Result act LeakyReLU:", a6) # 使用自定义的激活函数的结果
