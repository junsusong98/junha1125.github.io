---
layout: post
title: 【Pytorch 실습】 AutoEncoder를 사용한, Fashion Mnist data를 활용
description: > 
    아래의 코드는 Kaggle 및 Git의 공개된 코드를 적극 활용한, 과거의 공부한 내용을 정리한 내용입니다.  
---


Fashion MNIST train, test 데이터셋을 미리 다운로드 할 필요 없음.
PS.   
1. jupyter 실행결과도 코드 박스로 출력되어 있습니다.   
2. matplot을 이용해서 이미지를 출력하는 이미지는 첨부하지 않았습니다.  

## 1. Data 다운 받고 torch에 Load 하기.


```python
# torchvision 모듈을 이용해서 데이터 셋을 다운 받을 것이다. 
import torch
from torchvision import transforms, datasets  
BATCH_SIZE = 64

trainset = datasets.FashionMNIST(
    root      = './data/FASHIONMNIST/', 
    train     = True,
    download  = True,
    transform = transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(
    dataset     = trainset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    num_workers = 2
)
```

## 2. 모델 설계하기


```python
from torch import nn, optim

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        # 데이터 Feature 뽑기
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        # Feature를 이용해서 데이터 확장 해보기
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

model = AE().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr = 0.005)
criterion = nn.MSELoss()

print("Model: ", model)
print("Device: ", DEVICE)
```

    Model:  AE(
      (encoder): Sequential(
        (0): Linear(in_features=784, out_features=128, bias=True)
        (1): ReLU()
        (2): Linear(in_features=128, out_features=64, bias=True)
      )
      (decoder): Sequential(
        (0): Linear(in_features=64, out_features=128, bias=True)
        (1): ReLU()
        (2): Linear(in_features=128, out_features=784, bias=True)
      )
    )
    Device:  cuda



```python
# encoder decoder를 모두 통과한 후의 데이터를 출력해보기 위해서 다음과 같이 데이터 후처리
view_data = trainset.data[:5].view(-1, 28*28)
view_data = view_data.type(torch.FloatTensor) / 255.
```


```python
# Definite Train & Evaluate
def train(model, train_loader, optimizer):
    model.train()
    for step, (x, label) in enumerate(train_loader):
        x = x.view(-1, 28 * 28).to(DEVICE)
        y = x.view(-1, 28 * 28).to(DEVICE)
        label = label.to(DEVICE)
        
        encoded, decoded = model(x)
        # AutoEncoder를 지나서 자기자신이 되도록 학습된다. 
        # loss값을 y와 'decoded를 정규화한 후'의 값과의 차이로 구했다면 더 좋았을 것 같다. 
        loss = criterion(decoded, y)  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, step * len(x), len(train_loader.dataset), 100. * step / len(train_loader), loss.item()))
```

## 4. Train 시키기. Train 하면서 중간결과도 출력해보기


```python
''' Training'''
import numpy as np
import matplotlib.pyplot as plt

EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer)
    test_x = view_data.to(DEVICE)
    encoded_data, decoded_data = model(test_x)
    f, a = plt.subplots(2, 5, figsize = (10, 4))
    print("[Epoch {}]".format(epoch))

    # 원래 이미지의 사진 출력
    for idx in range(5):
        img = np.reshape(view_data.data.numpy()[idx], (28, 28))
        a[0][idx].imshow(img, cmap = "gray")
        a[0][idx].set_xticks(())
        a[0][idx].set_yticks(())
        
    # Encoder와 Decoder를 모두 통화한 후 사진 출력
    for idx in range(5):
        img = np.reshape(decoded_data.to("cpu").data.numpy()[idx], (28, 28))
        a[1][idx].imshow(img, cmap = "gray")
        a[1][idx].set_xticks(())
        a[1][idx].set_yticks(())
    plt.show()
```

## 5. 위에서 압축한 Feature를 가지고 Classification해보자!  
- encodered 데이터를 사용함으로 feature수가 작아지고, Classification 속도가 빨라짐을 알 수 있다.  
- 위에서는 Input과 동일한 Output을 만드는 Autoencoder를 설계했다.   
- 여기서는 Supervised Learnging을 할것이다. Fashion 옷의 Class가 뭔지 예측한다.    


```python
# 굳이 모델을 설계하지 않고도, lightgbm이라는 모듈을 사용해서 모델을 학습시킬 수 있다. 이런게 있구나.. 정도로만 알아두기.
# 여기서는 학습이 오래 걸린다. 

import time
import lightgbm as lgb
from sklearn.metrics import accuracy_score
start = time.time() 
lgb_dtrain = lgb.Dataset(data = trainset.train_data.view(-1, 28 * 28).numpy(), label = list(trainset.train_labels.numpy()))
lgb_param = {'max_depth': 10,
            'learning_rate': 0.001,
            'n_estimators': 20,
            'objective': 'multiclass',
            'num_class': len(set(list(trainset.train_labels.numpy()))) + 1}

num_round = 10000
lgb_model = lgb.train(params = lgb_param, num_boost_round = num_round, train_set = lgb_dtrain)              # 여기서 학습을 진행한다. 모델의 파라메터를 학습 완료!
lgb_model_predict = np.argmax(lgb_model.predict(trainset.train_data.view(-1, 28 * 28).numpy()), axis = 1)   # TestDataset은 없고, 그냥 Traindata로 Inference!
print("Accuracy: %.2f" % (accuracy_score(list(trainset.train_labels.numpy()), lgb_model_predict) * 100), "%") 
print("Time: %.2f" % (time.time() - start), "seconds")
```

    c:\users\justin\venv\lib\site-packages\torchvision\datasets\mnist.py:53: UserWarning: train_data has been renamed data
      warnings.warn("train_data has been renamed data")
    c:\users\justin\venv\lib\site-packages\torchvision\datasets\mnist.py:43: UserWarning: train_labels has been renamed targets
      warnings.warn("train_labels has been renamed targets")
    c:\users\justin\venv\lib\site-packages\lightgbm\engine.py:148: UserWarning: Found `n_estimators` in params. Will use it instead of argument
      warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))


    Accuracy: 82.84 %
    Time: 19.64 seconds



```python
# 여기서는 학습이 빨리 이뤄지는 것을 확인할 수 있다.
# 왜냐면, Encoder한 값, 즉 작은 Feature Map Data(784 -> 64)를 사용하기 때문이다.  
# 하지만 낮은 차원의 Feature를 이용해서 학습을 시키므로, 정확도가 떨어지는 것을 확인할 수 있다. 
 
train_encoded_x = trainset.train_data.view(-1, 28 * 28).to(DEVICE)
train_encoded_x = train_encoded_x.type(torch.FloatTensor)
train_encoded_x = train_encoded_x.to(DEVICE)
encoded_data, decoded_data = model(train_encoded_x) # 위에서 만든 모델로 추론한 결과를 아래의 학습에 사용한다! 
encoded_data = encoded_data.to("cpu")

start = time.time() 
lgb_dtrain = lgb.Dataset(data = encoded_data.detach().numpy(), label = list(trainset.train_labels.numpy()))
lgb_param = {'max_depth': 10,
            'learning_rate': 0.001,
            'n_estimators': 20,
            'objective': 'multiclass',
            'num_class': len(set(list(trainset.train_labels.numpy()))) + 1}

num_round = 10000
lgb_model = lgb.train(params = lgb_param, num_boost_round = num_round, train_set = lgb_dtrain)   # 여기서 학습을 진행한다. 모델의 파라메터를 학습 완료!
lgb_model_predict = np.argmax(lgb_model.predict(encoded_data.detach().numpy()), axis = 1)        # TestDataset은 없고, 그냥 Traindata로 Inference!
print("Accuracy: %.2f" % (accuracy_score(list(trainset.train_labels.numpy()), lgb_model_predict) * 100), "%") 
print("Time: %.2f" % (time.time() - start), "seconds") 
```

    Accuracy: 76.07 %
    Time: 1.96 seconds



```python

```
