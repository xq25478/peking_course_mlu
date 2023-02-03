# 第一课课上配套代码
import os
# os.environ['TL_BACKEND'] = 'torch'

# 导入TensorLayerX库
import  torch
from torch.nn import Linear
from torch.nn import ReLU,Softmax
from torch.nn import Sequential


torch.manual_seed(99999)  # set random set


# Single neuro
# 单个神经元

# matrix multiplication 矩阵乘法
x = torch.FloatTensor([[1., 2., 3.]]) # 1x3
w = torch.FloatTensor([[-0.5], [0.2], [0.1]]) # 3x1

print("X:\n", x, "\nShape:", x.shape)
print("W:\n", w, "\nShape", w.shape)

# Bias 偏置
b1 = torch.FloatTensor([0.5])

# 计算有偏置的矩阵乘法
z1 = torch.matmul(x, w)+b1

print("Z:\n", z1, "\nShape", z1.shape)

# Two outputs 两个输出

x = torch.FloatTensor([[1., 2., 3.]]) # 1x3

w = torch.FloatTensor([[-0.5, -0.3],
                           [0.2, 0.4],
                           [0.1, 0.15]]) # 3x2

print("X:\n", x, "\nShape:", x.shape)
print("W:\n", w, "\nShape", w.shape)

# Bias 偏置
b2 = torch.FloatTensor([0.5, 0.4])

z2 = torch.matmul(x, w)+b2

print("Z:\n", z2, "\nShape", z2.shape)


# Activation function 激活函数
# Reference: https://en.wikipedia.org/wiki/Activation_function
# Sigmoid function
sigmoid_torch = torch.nn.Sigmoid()
a1 = sigmoid_torch(z1)
print("Result torch sigmoid:", a1)

# define your own activation function


class ActSigmoid(torch.nn.Module):
    def forward(self, x):
        return 1 / (1 + torch.exp(-x))


sigmoid_act = ActSigmoid()
a1_m = sigmoid_act(z1)

print("Result your own sigmoid:", a1_m)

# Softmax
a2 = torch.softmax(z2, axis=-1)
print("Result torch softmax:", a2)


class ActSoftmax(torch.nn.Module):
    def forward(self, x, axis=-1):
        # e = torch.exp(x - torch.reduce_max(x, axis=axis, keepdims=True))
        e = torch.exp(x - torch.max(x, dim=axis, keepdims=True).values)
        s = torch.sum(e, dim=axis, keepdims=True)
        return e / s


softmax_act = ActSoftmax()
a2_m = softmax_act(z2, axis=-1)
print("Result your own softmax:", a2_m)

MLP = Sequential(
Linear(out_features=3, in_features=3 ),
ReLU(),
Linear(out_features=3, in_features=3),
ReLU(),
Linear(out_features=3, in_features=3),
ReLU(),
Linear(out_features=2, in_features=3),
Softmax()
)

out = MLP(x)
print("Neural network output: ", out.shape)

# # Get parameters of the model
# # 获取模型参数
all_weights_generator = MLP.parameters()

linear1_weights = MLP[0].weight # MLP[0].all_weights
# # Loss functions 损失函数
# # Reference: https://en.wikipedia.org/wiki/Loss_function

# Mean absolute error 平均绝对误差
def mae(output, target):
    return torch.mean(torch.abs(output-target), dim=-1)

# Mean squared error 平均平方误差
def mse(output, target):
    return torch.mean(torch.square(output-target), dim=-1)


y1 = torch.FloatTensor([1., 3., 5., 7.])
y2 = torch.FloatTensor([2., 4., 6., 8.])

# 计算MAE和MSE
l_mae = mae(y1, y2)
l_mse = mse(y1, y2)

print("Loss MAE: {} \nLoss MSE: {}".format(l_mae, l_mse))

# l_mse_torch = torch.losses.mean_squared_error(y1, y2)
l_mse_torch = torch.nn.MSELoss()(y1, y2)
print("Loss MSE by torch: {}".format(l_mse_torch.numpy()))

# 二分类标签
target_binary = torch.FloatTensor([[1., 0.]])

print("Target value:{}".format(target_binary.numpy()))
print("Neural network output:{}".format(out.detach().numpy()))

# Binary cross entropy 二分类交叉熵
# l_bce = torch.losses.binary_cross_entropy(out, target_binary)
l_bce = torch.nn.BCEWithLogitsLoss()(out, target_binary)
print("Loss binary cross entropy:{}".format(l_bce.detach().numpy()))


# Error Back-Propagation 误差梯度反向传播
# 使用自动求导机制
# x = torch.Variable(torch.FloatTensor(1.,requires_grad=True))
# x.stop_gradient = False
# w = torch.Variable(torch.FloatTensor(0.5,requires_grad=True))
# w.stop_gradient = False

device = torch.device("cpu") # TODO change to mlu
# x = torch.FloatTensor([1.],device=device,requires_grad=True)
# w = torch.FloatTensor([0.5],device=device,requires_grad=True)
x = torch.FloatTensor([1.],device=device)
x.requires_grad=True
w = torch.FloatTensor([0.5],device=device)
w.requires_grad=True

# 前向传播
t = x+w
z = t**2

# 反向传播
z.backward()

print("Gradient of w is: {}".format(w.grad.numpy()))

# BP for the network
l_bce.backward()


# TODO write out grad
print("Gradient of the network layer 1's weights is: \n{}".format(
    MLP[0].weight.grad.numpy()))

#Optimization 优化
# opt = torch.optimizers.SGD(lr=0.01, momentum=0.9) # 随机梯度下降优化器
opt = torch.optim.SGD(MLP.parameters(),lr=0.01, momentum=0.9) # 随机梯度下降优化器
weights = list(MLP.parameters())
# for param in weights:
#     print("===========================")
#     print(param)

x_in = torch.FloatTensor([[1., 2., 3.]])
target = torch.FloatTensor([[1., 0.]])

# print("Before optimization, network layer 1's weights is: \n{}".format(
#     MLP.layer_list[0].weights.detach().numpy()))
print("Before optimization, network layer 1's weights is: \n{}".format(
    MLP[0].weight.detach().numpy()))

out = MLP(x_in) # 前向传播
l_bce = torch.nn.BCEWithLogitsLoss()(out, target) # 计算损失

# grads = opt.gradient(l_bce, MLP.trainable_weights) # 计算梯度
# opt.apply_gradients(zip(grads, MLP.trainable_weights)) # 更新参数
l_bce.backward()# 计算梯度
opt.step()# 更新参数
print("After optimization, network layer 1's weights is: \n{}".format(
    MLP[0].weight.detach().numpy()))


