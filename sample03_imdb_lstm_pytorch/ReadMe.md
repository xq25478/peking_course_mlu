# ex1

## run tensorlayerX demo
```
python ex1_imdb_lstm.py
```

log
```
Epoch 1 of 5 took 0.01702260971069336
   train loss: 0.6931127905845642
   train acc:  0.546875

....

Epoch 5 of 5 took 2.6423957347869873
   train loss: 0.17205196619033813
   train acc:  0.9357527955271565
```


## run mlu torch demo

```
python  ex1_imdb_lstm_torch.py
```

log
```
vocab_size:  20001
ImdbNet(
  (embedding): Embedding(20001, 64)
  (lstm): LSTM(64, 64)
  (linear1): Linear(in_features=64, out_features=64, bias=True)
  (act1): ReLU()
  (linear2): Linear(in_features=64, out_features=2, bias=True)
)
Train Epoch: 1 Loss: 0.636723    Acc: 0.620707
Train Epoch: 2 Loss: 0.421770    Acc: 0.809355
Train Epoch: 3 Loss: 0.328943    Acc: 0.860972
Train Epoch: 4 Loss: 0.270409    Acc: 0.890625
Train Epoch: 5 Loss: 0.229245    Acc: 0.911292
```