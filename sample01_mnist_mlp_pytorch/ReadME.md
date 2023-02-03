# ex1

tensorlayerX
```
python ex1_activations.py
```

```log
Using PyTorch backend.
/torch/venv3/pytorch/lib/python3.7/site-packages/torch/nn/functional.py:1810: UserWarning: nn.functional.tanh is deprecated. Use torch.tanh instead.
  warnings.warn("nn.functional.tanh is deprecated. Use torch.tanh instead.")
Z:
 tensor([[0.7000]]) 
Shape torch.Size([1, 1])
Result tlx sigmoid: tensor([[0.6682]])
Result act sigmoid: tensor([[0.6682]])
Result tlx Tanh: tensor([[0.6044]])
Result act Tanh: tensor([[0.6044]])
Result tlx LeakyReLU: tensor([[0.7000]])
Result act LeakyReLU: tensor([[0.7000]])
```

mlu torch
```
python ex1_activations_torch.py
```

```log
Z:
 tensor([[0.7000]], device='mlu:0') 
Shape torch.Size([1, 1])
Result torch sigmoid: tensor([[0.6682]], device='mlu:0')
Result act sigmoid: tensor([[0.6682]], device='mlu:0')
Result torch Tanh: tensor([[0.6044]], device='mlu:0')
Result act Tanh: tensor([[0.6044]], device='mlu:0')
Result torch LeakyReLU: tensor([[0.7000]], device='mlu:0')
Result act LeakyReLU: tensor([[0.7000]], device='mlu:0')
```


# ex2


run tensorlayerX
```
python ex2_mnist_mlp.py
```

log
```
[2023-2-3 10:27:19] [CNNL] [Warning]: [cnnlClip] is deprecated and will be removed in the future release, please use [cnnlClip_v2] instead.
Train data shape:(50000, 784)
Start training

Epoch 1 of 50 took 1.3772239685058594
   train loss: 0.28411951661109924
   train acc:  0.9169317455242967
Epoch 2 of 50 took 1.379014492034912
   train loss: 0.15157920122146606
   train acc:  0.9563738810741688
Epoch 3 of 50 took 1.3680264949798584
   train loss: 0.13018009066581726
   train acc:  0.9637068414322251
Epoch 4 of 50 took 1.3828151226043701
   train loss: 0.10864217579364777
   train acc:  0.969725063938619
Epoch 5 of 50 took 1.3816955089569092
   train loss: 0.09308739006519318
   train acc:  0.974528452685422
Epoch 6 of 50 took 1.369121789932251
   train loss: 0.08843890577554703
   train acc:  0.9759071291560102
Epoch 7 of 50 took 1.366574764251709
   train loss: 0.08682134747505188
   train acc:  0.9759390984654732
Epoch 8 of 50 took 1.3613207340240479
   train loss: 0.0819404125213623
   train acc:  0.9776694373401534
Epoch 9 of 50 took 1.3623497486114502
   train loss: 0.08432600647211075
   train acc:  0.9776614450127877
Epoch 10 of 50 took 1.369225025177002
   train loss: 0.07463758438825607
   train acc:  0.9812579923273658
Epoch 11 of 50 took 1.363034725189209
   train loss: 0.0737389326095581
   train acc:  0.9816815856777493
Epoch 12 of 50 took 1.3727574348449707
   train loss: 0.06547487527132034
   train acc:  0.9837515984654732
Epoch 13 of 50 took 1.3640925884246826
   train loss: 0.0661378800868988
   train acc:  0.9830242966751919
Epoch 14 of 50 took 1.3586909770965576
   train loss: 0.05339909344911575
   train acc:  0.9854819373401534
Epoch 15 of 50 took 1.3588767051696777
   train loss: 0.05922103673219681
   train acc:  0.9845828005115089
Epoch 16 of 50 took 1.3630759716033936
   train loss: 0.058702196925878525
   train acc:  0.9848984974424552
Epoch 17 of 50 took 1.3677616119384766
   train loss: 0.05809531733393669
   train acc:  0.9853420716112532
Epoch 18 of 50 took 1.3624634742736816
   train loss: 0.056517623364925385
   train acc:  0.985817615089514
Epoch 19 of 50 took 1.3616559505462646
   train loss: 0.05512915551662445
   train acc:  0.987116368286445
Epoch 20 of 50 took 1.3644118309020996
   train loss: 0.05458595231175423
   train acc:  0.9864729859335039
Epoch 21 of 50 took 1.3651225566864014
   train loss: 0.04544315114617348
   train acc:  0.9882592710997443
Epoch 22 of 50 took 1.366999864578247
   train loss: 0.05900931358337402
   train acc:  0.986528932225064
Epoch 23 of 50 took 1.3672616481781006
   train loss: 0.04884298890829086
   train acc:  0.9880115089514067
Epoch 24 of 50 took 1.3638112545013428
   train loss: 0.04601839929819107
   train acc:  0.9890505115089514
Epoch 25 of 50 took 1.366443395614624
   train loss: 0.05118713527917862
   train acc:  0.9880075127877238
Epoch 26 of 50 took 1.3625028133392334
   train loss: 0.05355485528707504
   train acc:  0.9880115089514067
Epoch 27 of 50 took 1.376105546951294
   train loss: 0.05006958916783333
   train acc:  0.988618925831202
Epoch 28 of 50 took 1.3709983825683594
   train loss: 0.03550583869218826
   train acc:  0.9912563938618926
Epoch 29 of 50 took 1.3715870380401611
   train loss: 0.032423943281173706
   train acc:  0.9917279411764706
Epoch 30 of 50 took 1.3742120265960693
   train loss: 0.03863848000764847
   train acc:  0.990828804347826
Epoch 31 of 50 took 1.3727049827575684
   train loss: 0.05018218234181404
   train acc:  0.9889066496163683
Epoch 32 of 50 took 1.3723626136779785
   train loss: 0.052130378782749176
   train acc:  0.9890625
Epoch 33 of 50 took 1.371131181716919
   train loss: 0.03217719495296478
   train acc:  0.9924952046035805
Epoch 34 of 50 took 1.376847267150879
   train loss: 0.0298140998929739
   train acc:  0.9929068094629157
Epoch 35 of 50 took 1.3726122379302979
   train loss: 0.05672585964202881
   train acc:  0.9878516624040921
Epoch 36 of 50 took 1.3740558624267578
   train loss: 0.06260025501251221
   train acc:  0.9884510869565217
Epoch 37 of 50 took 1.374136209487915
   train loss: 0.037057094275951385
   train acc:  0.9920835997442455
Epoch 38 of 50 took 1.372375249862671
   train loss: 0.034384943544864655
   train acc:  0.9921675191815856
Epoch 39 of 50 took 1.3703250885009766
   train loss: 0.024515073746442795
   train acc:  0.9938738810741689
Epoch 40 of 50 took 1.371016263961792
   train loss: 0.04346372187137604
   train acc:  0.9909886508951407
Epoch 41 of 50 took 1.3719110488891602
   train loss: 0.04218772053718567
   train acc:  0.9914002557544757
Epoch 42 of 50 took 1.370954990386963
   train loss: 0.03694915398955345
   train acc:  0.9921954923273658
Epoch 43 of 50 took 1.3709664344787598
   train loss: 0.02828783541917801
   train acc:  0.9934023337595909
Epoch 44 of 50 took 1.3703968524932861
   train loss: 0.0404810830950737
   train acc:  0.9917559143222506
Epoch 45 of 50 took 1.370628833770752
   train loss: 0.03955986723303795
   train acc:  0.9923833120204604
Epoch 46 of 50 took 1.3718085289001465
   train loss: 0.03774384409189224
   train acc:  0.9927149936061381
Epoch 47 of 50 took 1.371767282485962
   train loss: 0.027404066175222397
   train acc:  0.9938818734015346
Epoch 48 of 50 took 1.3774938583374023
   train loss: 0.021360665559768677
   train acc:  0.9948449488491049
Epoch 49 of 50 took 1.3725638389587402
   train loss: 0.046228520572185516
   train acc:  0.9913363171355499
Epoch 50 of 50 took 1.3721795082092285
   train loss: 0.029009182006120682
   train acc:  0.993486253196931
Epoch progress 50/50   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 0:01:08
Batch progress 391/391                                            0% 0:00:01 0:00:00
```

run ex2 mlu torch demo
```
python ex2_mnist_mlp_torch.py
```


log
```
Train Epoch: 1 Loss: 0.311231   Acc: 0.909981
Train Epoch: 2 Loss: 0.161377   Acc: 0.953997
Train Epoch: 3 Loss: 0.129204   Acc: 0.963991
Train Epoch: 4 Loss: 0.125882   Acc: 0.965740
Train Epoch: 5 Loss: 0.099271   Acc: 0.972565
Train Epoch: 6 Loss: 0.088590   Acc: 0.976141
Train Epoch: 7 Loss: 0.093701   Acc: 0.974769
Train Epoch: 8 Loss: 0.086585   Acc: 0.976274
Train Epoch: 9 Loss: 0.079104   Acc: 0.977418
Train Epoch: 10 Loss: 0.084387  Acc: 0.977345
Train Epoch: 11 Loss: 0.070697  Acc: 0.980616
Train Epoch: 12 Loss: 0.072877  Acc: 0.980671
Train Epoch: 13 Loss: 0.072293  Acc: 0.980838
Train Epoch: 14 Loss: 0.063755  Acc: 0.982870
Train Epoch: 15 Loss: 0.069826  Acc: 0.982704
Train Epoch: 16 Loss: 0.064564  Acc: 0.983670
Train Epoch: 17 Loss: 0.052840  Acc: 0.985885
Train Epoch: 18 Loss: 0.057868  Acc: 0.985002
Train Epoch: 19 Loss: 0.049400  Acc: 0.987301
Train Epoch: 20 Loss: 0.060471  Acc: 0.984947
Train Epoch: 21 Loss: 0.059583  Acc: 0.985419
Train Epoch: 22 Loss: 0.063792  Acc: 0.984786
Train Epoch: 23 Loss: 0.049971  Acc: 0.987584
Train Epoch: 24 Loss: 0.048933  Acc: 0.987934
Train Epoch: 25 Loss: 0.047283  Acc: 0.988251
Train Epoch: 26 Loss: 0.049169  Acc: 0.988334
Train Epoch: 27 Loss: 0.057065  Acc: 0.986818
Train Epoch: 28 Loss: 0.046641  Acc: 0.988684
Train Epoch: 29 Loss: 0.041114  Acc: 0.990016
Train Epoch: 30 Loss: 0.042195  Acc: 0.989533
Train Epoch: 31 Loss: 0.044245  Acc: 0.989283
Train Epoch: 32 Loss: 0.052061  Acc: 0.988301
Train Epoch: 33 Loss: 0.045482  Acc: 0.988967
Train Epoch: 34 Loss: 0.041573  Acc: 0.989350
Train Epoch: 35 Loss: 0.043013  Acc: 0.990416
Train Epoch: 36 Loss: 0.045333  Acc: 0.989867
Train Epoch: 37 Loss: 0.040720  Acc: 0.990366
Train Epoch: 38 Loss: 0.031698  Acc: 0.992382
Train Epoch: 39 Loss: 0.050066  Acc: 0.989333
Train Epoch: 40 Loss: 0.052609  Acc: 0.988934
Train Epoch: 41 Loss: 0.040010  Acc: 0.991033
Train Epoch: 42 Loss: 0.030896  Acc: 0.993065
Train Epoch: 43 Loss: 0.039474  Acc: 0.991216
Train Epoch: 44 Loss: 0.055124  Acc: 0.989267
Train Epoch: 45 Loss: 0.036554  Acc: 0.991799
Train Epoch: 46 Loss: 0.030485  Acc: 0.992648
Train Epoch: 47 Loss: 0.040328  Acc: 0.991699
Train Epoch: 48 Loss: 0.040726  Acc: 0.991166
Train Epoch: 49 Loss: 0.032827  Acc: 0.992615
Train Epoch: 50 Loss: 0.052154  Acc: 0.990094
```

# ex3
只用了numpy 所以没有修改


# examples


run tensorlayerX
```
python examples.py
```

log
```
Using PyTorch backend.
X:
 tensor([[1., 2., 3.]]) 
Shape: torch.Size([1, 3])
W:
 tensor([[-0.5000],
        [ 0.2000],
        [ 0.1000]]) 
Shape torch.Size([3, 1])
Z:
 tensor([[0.7000]]) 
Shape torch.Size([1, 1])
X:
 tensor([[1., 2., 3.]]) 
Shape: torch.Size([1, 3])
W:
 tensor([[-0.5000, -0.3000],
        [ 0.2000,  0.4000],
        [ 0.1000,  0.1500]]) 
Shape torch.Size([3, 2])
Z:
 tensor([[0.7000, 1.3500]]) 
Shape torch.Size([1, 2])
Result tlx sigmoid: tensor([[0.6682]])
Result your own sigmoid: tensor([[0.6682]])
Result tlx softmax: tensor([[0.3430, 0.6570]])
Result your own softmax: tensor([[0.3430, 0.6570]])
/torch/venv3/pytorch/lib/python3.7/site-packages/tensorlayerx/backend/ops/torch_nn.py:339: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.softmax(x, dim=self.axis)
Neural network output:  torch.Size([1, 2])
Loss MAE: 1.0 
Loss MSE: 1.0
Loss MSE by TLX: 1.0
Target value:[[1. 0.]]
Neural network output:[[0.5 0.5]]
Loss binary cross entropy:0.6931471824645996
Gradient of w is: 3.0
Gradient of the network layer 1's weights is: 
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]
Before optimization, network layer 1's weights is: 
[[-0.00263621 -0.01939446  0.0005258 ]
 [-0.02377951  0.01863976 -0.01198454]
 [ 0.01019265 -0.01377996  0.00451451]]
After optimization, network layer 1's weights is: 
[[-0.00263621 -0.01939446  0.0005258 ]
 [-0.02377951  0.01863976 -0.01198454]
 [ 0.01019265 -0.01377996  0.00451451]]
```


run mlu torch exmaples
```
python examples_torch.py
```
log
```
X:
 tensor([[1., 2., 3.]]) 
Shape: torch.Size([1, 3])
W:
 tensor([[-0.5000],
        [ 0.2000],
        [ 0.1000]]) 
Shape torch.Size([3, 1])
Z:
 tensor([[0.7000]]) 
Shape torch.Size([1, 1])
X:
 tensor([[1., 2., 3.]]) 
Shape: torch.Size([1, 3])
W:
 tensor([[-0.5000, -0.3000],
        [ 0.2000,  0.4000],
        [ 0.1000,  0.1500]]) 
Shape torch.Size([3, 2])
Z:
 tensor([[0.7000, 1.3500]]) 
Shape torch.Size([1, 2])
Result torch sigmoid: tensor([[0.6682]])
Result your own sigmoid: tensor([[0.6682]])
Result torch softmax: tensor([[0.3430, 0.6570]])
Result your own softmax: tensor([[0.3430, 0.6570]])
/torch/venv3/pytorch/lib/python3.7/site-packages/torch/nn/modules/container.py:139: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
  input = module(input)
Neural network output:  torch.Size([1, 2])
Loss MAE: 1.0 
Loss MSE: 1.0
Loss MSE by torch: 1.0
Target value:[[1. 0.]]
Neural network output:[[0.423719 0.576281]]
Loss binary cross entropy:0.7629010677337646
Gradient of w is: [3.]
Gradient of the network layer 1's weights is: 
[[0.00496678 0.00993355 0.01490033]
 [0.01376176 0.02752351 0.04128527]
 [0.         0.         0.        ]]
Before optimization, network layer 1's weights is: 
[[-0.5669673   0.27330336  0.54320264]
 [ 0.33784539  0.22147375  0.25147405]
 [ 0.14897908 -0.46751714 -0.03618013]]
After optimization, network layer 1's weights is: 
[[-0.56706667  0.2731047   0.5429046 ]
 [ 0.33757016  0.22092329  0.25064835]
 [ 0.14897908 -0.46751714 -0.03618013]]
```