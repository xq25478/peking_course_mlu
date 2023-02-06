基于MLU370的lstm 文本生成示例程序

### 训练 && 验证 
- 运行(模型开始训练并会对给出的一句话进行后续文本生成)
```
bash train.sh
```
- 结果
```
{'epoch': 499, 'batch': 84, 'loss': 0.023613790050148964}
{'epoch': 499, 'batch': 85, 'loss': 0.025516167283058167}
{'epoch': 499, 'batch': 86, 'loss': 0.04155672341585159}
{'epoch': 499, 'batch': 87, 'loss': 0.02164582535624504}
{'epoch': 499, 'batch': 88, 'loss': 0.023728517815470695}
{'epoch': 499, 'batch': 89, 'loss': 0.026620715856552124}
{'epoch': 499, 'batch': 90, 'loss': 0.04648686572909355}
{'epoch': 499, 'batch': 91, 'loss': 0.023265302181243896}
{'epoch': 499, 'batch': 92, 'loss': 0.023684531450271606}
{'epoch': 499, 'batch': 93, 'loss': 0.018991224467754364}
#下面一行为对输入的话进行后续文本预测
['Knock', 'knock.', 'Whos', 'there?', 'use', 'the', 'cow', 'say', 'into', 'the', 'jumper', 'Frank', 'that', "doesn't", 'not', 'side', 'Did', 'you', 'hear', 'about', 'the', 'two', 'silk', 'worms', 'that', 'wants', 'to', 'start', '&gt;', 'Robert', 'need', 'in', 'start', 'car?', 'It', 'which', 'tribesman', 'said', 'there', 'was', 'so', 'Mom', 'single', 'far,', 'but', 'of', 'them', 'far', 'who', 'said', 'he', 'wet', 'it', 'scares', 'your', 'best', 'joke', 'since', 'I', 'was', 'thinking', 'of', 'them', 'today', 'steal', 'me', 'before', 'me', 'students', 'to', 'said,', 'Why', 'were', 'the', 'German', 'Horseman', 'blush?', 'It', 'was', 'such', 'a', 'little', 'wine', 'What', 'do', 'cows', 'unlock', 'a', 'monastery', 'key', 'that', 'opens', 'come', 'tie', 'coming', 'around?', 'Why', 'did', 'the', 'chicken', 'cross', 'the', 'road?', '...he']

```