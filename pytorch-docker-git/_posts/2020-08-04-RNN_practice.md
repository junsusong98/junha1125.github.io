---
layout: post
title: 【Pytorch 실습】RNN, LSTM을 활용한 영화 평점 예측 모델.
description: > 
        아래의 코드는 Kaggle 및 Git의 공개된 코드를 적극 활용한, 과거의 공부한 내용을 정리한 내용입니다.  
---

【Pytorch 실습】RNN, LSTM을 활용한 영화 리뷰 예측 모델.
{:.lead}
- torchvision이 아니라 torchtext에서 데이터를 가져오기 위해 torchtext import  
- 자연어 같은 경우 Text의 전처리가 가장 중요하다.  
- 자연어를 다뤄야할 일이 있다면 아래의 내용이나 torch document의 내용을 참고 공부하고 공부한다면 더 좋을 것 같다. 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets
```

```python
BATCH_SIZE = 100
LR = 0.001
EPOCHS = 15
USE_CUDA = torch.cuda.is_available()
```

## 1. Data Load(Text이므로 어려울 수 있다.)

- torchtext.data 에서 데이터를 가져온다. Field(어떻게 데이터를 처리하겠다. 라는 것을 정의하는 함수)라는 데이터를 가져올 것이다.
- [https://torchtext.readthedocs.io/en/latest/data.html#field](https://torchtext.readthedocs.io/en/latest/data.html#field)

```python
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
TEXT = data.Field(sequential = True, batch_first = True, lower = True) # lower : 전부 소문자로
LABEL = data.Field(sequential = False, batch_first = True) # 평점 : 긍정 vs 부정
```

- datasets들 중에서 IMDB 데이터를 가져온다. 영화 리뷰 글(Text) + 긍정/부정(Lavel)가 있다. 
```python
trainset, testset = datasets.IMDB.splits(TEXT, LABEL) 
```
- 데이터가 문장이 되어 있으므로, 띄어쓰기 단위로 단어로 자른다. 그런 후 단어가 총 몇개가 있는지 확인한다. Ex i am a boy = 4 개
```python
TEXT.build_vocab(trainset, min_freq = 5) # 최고 5번 이상 나온 단어만 voca로 구분하겠다. 
LABEL.build_vocab(trainset)
```

- Train dataset과 valuation dataset으로 구분한다. split 함수를 사용하면 쉽다.
- iter로 놓음으로써, 나중에 베치사이즈 기준으로 학습 시키기 쉽게 만든다. 
```python
trainset, valset = trainset.split(split_ratio = 0.8)
train_iter, val_iter, test_iter = data.BucketIterator.splits((trainset, valset, testset), batch_size = BATCH_SIZE, shuffle = True, repeat = False)
```

```python
vocab_size = len(TEXT.vocab)
n_classes = 2  # 긍정 vs 부정
print("[TRAIN]: %d \t [VALID]: %d \t [TEST]: %d \t [VOCAB] %d \t [CLASSES] %d" % (len(trainset), len(valset), len(testset), vocab_size, n_classes))
```

[TRAIN]: 20000 [VALID]: 5000 [TEST]: 25000 
[VOCAB] 46159(Train set안에 있는 voca만) [CLASSES] 2

## 2. RNN 모델 구현

```python
class BasicRNN(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p = 0.2):
        super(BasicRNN, self).__init__()
        print("Building RNN")
        self.n_layers = n_layers # layer 갯수 RNN을 몇번 돌건지
        self.embed = nn.Embedding(n_vocab, embed_dim) # 한 단어를 하나의 백터/값으로 임베팅 한다.
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.RNN(embed_dim, self.hidden_dim, num_layers = self.n_layers, batch_first = True) # nn.RNN 을 이용하면 쉽게 RNN구현 가능!!
        self.out = nn.Linear(self.hidden_dim, n_classes) 

    def forward(self, x):
        x = self.embed(x) # 문자를 숫자/백터로 변환
        h_0 = self._init_state(batch_size = x.size(0)) # 가장 첫번째 h0는 아직 정의되지 않았으므로 다음과 같이 정의 해준다. 
        x, _ = self.rnn(x, h_0) # 이렇게 손쉽게 RNN을 구현할 수 있다. 
        h_t = x[:, -1, :] # 모든 문장을 거쳐서 나온 가장 마지막에 나온 단어(평점)의 값
        self.dropout(h_t)
        logit = torch.sigmoid(self.out(h_t))
        return logit

    def _init_state(self, batch_size = 1):
        weight = next(self.parameters()).data # hidden state에 대한 차원은 맞춰주면서 가중치 값은 아직은 0으로 만들어 준다. 
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
```

## 3. 학습 및 추론

```python
def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1)
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()

        if b % 50 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(e,
                                                                           b * len(x),
                                                                           len(train_iter.dataset),
                                                                           100. * b / len(train_iter),
                                                                           loss.item()))
```

```python
def evaluate(model, val_iter):
    model.eval()
    corrects, total_loss = 0, 0

    for batch in val_iter:
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1) # 0아니면 1로 데이터 값 수정
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction = "sum")
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()

    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy
```

```python
model = BasicRNN(n_layers = 1, hidden_dim = 256, n_vocab = vocab_size, embed_dim = 128, n_classes = n_classes, dropout_p = 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
for e in range(1, EPOCHS + 1):
    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, val_iter)
    print("[EPOCH: %d], Validation Loss: %5.2f | Validation Accuracy: %5.2f" % (e, val_loss, val_accuracy))
```


    Building RNN
    Train Epoch: 1 [0/20000 (0%)]	Loss: 0.693486
    Train Epoch: 1 [5000/20000 (25%)]	Loss: 0.693027
    Train Epoch: 1 [10000/20000 (50%)]	Loss: 0.692615
    Train Epoch: 1 [15000/20000 (75%)]	Loss: 0.693534
    [EPOCH: 1], Validation Loss:  0.69 | Validation Accuracy: 49.00

    Train Epoch: 15 [0/20000 (0%)]	Loss: 0.692949
    Train Epoch: 15 [5000/20000 (25%)]	Loss: 0.695515
    Train Epoch: 15 [10000/20000 (50%)]	Loss: 0.692522
    Train Epoch: 15 [15000/20000 (75%)]	Loss: 0.693040
    [EPOCH: 15], Validation Loss:  0.69 | Validation Accuracy: 50.00



```python
test_loss, test_acc = evaluate(model,test_iter)
print("Test Loss: %5.2f | Test Accuracy: %5.2f" % (test_loss, test_acc))
```

    Test Loss:  0.70 | Test Accuracy: 50.00

## 4. GRU 모델 구현 및 학습추론
- RNN과 전체적인 구조가 같으나, Hidden state에서 사용되는 gate가 2개이다. 
```python
class BasicGRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p = 0.2):
        super(BasicGRU, self).__init__()
        print("Building GRU")
        self.n_layers = n_layers
        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim, self.hidden_dim, num_layers = self.n_layers, batch_first = True)
        self.out = nn.Linear(self.hidden_dim, n_classes)
        
    def forward(self, x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size = x.size(0))
        x, _ = self.gru(x, h_0)
        h_t = x[:, -1, :]
        self.dropout(h_t)
        logit = torch.sigmoid(self.out(h_t))
        return logit
    
    def _init_state(self, batch_size = 1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
```


```python
model = BasicGRU(n_layers = 1, hidden_dim = 256, n_vocab = vocab_size, embed_dim = 128, n_classes = n_classes, dropout_p = 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
for e in range(1, EPOCHS + 1):
    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, val_iter)
    print("[EPOCH: %d], Validation Loss: %5.2f | Validation Accuracy: %5.2f" % (e, val_loss, val_accuracy))
```

    Building GRU
    Train Epoch: 1 [0/20000 (0%)]	Loss: 0.691595
    Train Epoch: 1 [5000/20000 (25%)]	Loss: 0.693059
    Train Epoch: 1 [10000/20000 (50%)]	Loss: 0.693562
    Train Epoch: 1 [15000/20000 (75%)]	Loss: 0.693485
    [EPOCH: 1], Validation Loss:  0.69 | Validation Accuracy: 50.00
 
    Train Epoch: 15 [0/20000 (0%)]	Loss: 0.363591
    Train Epoch: 15 [5000/20000 (25%)]	Loss: 0.324579
    Train Epoch: 15 [10000/20000 (50%)]	Loss: 0.335097
    Train Epoch: 15 [15000/20000 (75%)]	Loss: 0.333244
    [EPOCH: 15], Validation Loss:  0.45 | Validation Accuracy: 85.00



```python
test_loss, test_acc = evaluate(model,test_iter)
print("Test Loss: %5.2f | Test Accuracy: %5.2f" % (test_loss, test_acc))
```

    Test Loss:  0.46 | Test Accuracy: 84.00

## 5. GRU 모델 구현 및 학습추론
- 알고 있듯이 LSTM의 Cell state의 gate는 forget,input,ouput이 존재한다.
- 이렇게 nn모듈을 사용해서 LSTM을 쉽게 구현하고 사용할 수 있다. 


```python
class BasicLSTM(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p = 0.2):
        super(BasicLSTM, self).__init__()
        print("Building LSTM")
        self.n_layers = n_layers
        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(embed_dim, self.hidden_dim, num_layers = self.n_layers, batch_first = True)
        self.out = nn.Linear(self.hidden_dim, n_classes)
        
    def forward(self, x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size = x.size(0))
        c_0 = self._init_state(batch_size = x.size(0))
        
        x, _ = self.lstm(x, (h_0, c_0))
        h_t = x[:, -1, :]
        self.dropout(h_t)
        logit = torch.sigmoid(self.out(h_t))
        return logit
    
    def _init_state(self, batch_size = 1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
```


```python
model = BasicLSTM(n_layers = 1, hidden_dim = 256, n_vocab = vocab_size, embed_dim = 128, n_classes = n_classes, dropout_p = 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
for e in range(1, EPOCHS + 1):
    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, val_iter)
    print("[EPOCH: %d], Validation Loss: %5.2f | Validation Accuracy: %5.2f" % (e, val_loss, val_accuracy))
```

    Building LSTM
    Train Epoch: 1 [0/20000 (0%)]	Loss: 0.693381
    Train Epoch: 1 [5000/20000 (25%)]	Loss: 0.693180
    Train Epoch: 1 [10000/20000 (50%)]	Loss: 0.693065
    Train Epoch: 1 [15000/20000 (75%)]	Loss: 0.695363
    [EPOCH: 1], Validation Loss:  0.69 | Validation Accuracy: 49.00
   
    Train Epoch: 15 [0/20000 (0%)]	Loss: 0.427247
    Train Epoch: 15 [5000/20000 (25%)]	Loss: 0.454655
    Train Epoch: 15 [10000/20000 (50%)]	Loss: 0.428639
    Train Epoch: 15 [15000/20000 (75%)]	Loss: 0.428214
    [EPOCH: 15], Validation Loss:  0.49 | Validation Accuracy: 82.00



```python
test_loss, test_acc = evaluate(model,test_iter)
print("Test Loss: %5.2f | Test Accuracy: %5.2f" % (test_loss, test_acc))
```

    Test Loss:  0.50 | Test Accuracy: 80.00

