---
layout: post
title: 【Pytorch 실습】Pytorch 내장 Model 사용, 내장 weight 사용해서 Transfer Learning하기 
description: > 
        아래의 코드는 Kaggle 및 Git의 공개된 코드를 적극 활용한, 과거의 공부한 내용을 정리한 내용입니다.  
---
【Pytorch 실습】Pytorch 내장 Model 사용, 내장 weight 사용해서 Transfer Learning하기 

## 1. 데이터 로드
- 지금까지와는 약간 다르게 데이터 로드
- hymenoptera_data를 사용할 예정이다. [다운로드](https://www.kaggle.com/ajayrana/hymenoptera-data) 는 여기서 하면 되고, 데이터 구조는 다음과 같다. 꿀벌과 개미를 분류하기 위한 데이터 셋이다.   
    ![image](https://user-images.githubusercontent.com/46951365/91165408-89c0ec80-e70b-11ea-81d1-a51e9564c1e1.png)
- 아래와 같은 방법 전처리 할 수 있다. 지금까지의 dataset, dataloader를 구성할 때와의 방법과는 조금 다르다. 
- 적은 데이터를 이용하기 위해, Data argumentation하는 방법이 아래와 같이 transforms.Compose를 이용하면 되는 것을 알아두자. 


```python
import torch 
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
```


```python
from torchvision import datasets, transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder("../data/hymenoptera_data", data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 8, num_workers = 0, shuffle = True) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
```

## 2. 학습 및 추론 함수 정의  

```python
def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, 
                batch_idx * len(data), 
                len(train_loader.dataset), 
                100. * batch_idx / len(train_loader), 
                loss.item()))


def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction = "sum").item()
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy
```

## 3. 내장 모델 사용하기(Transfer Learning 안함)
- pretrained = True 를 사용하면 ImageNet에서 사용한 파라메터를 사용한다. 
- torch의 내장 모델을 그냥 사용할 때, input과 output사이즈(Groud True data size)는 현재 나의 data에 따라서 달라지기 떄문에 전혀 걱정하지 않아도 된다.   

```python
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

model = models.resnet18(pretrained = False).cuda()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    train(model, dataloaders["train"], optimizer, epoch)
    test_loss, test_accuracy = evaluate(model, dataloaders["val"])
    print("[{}] Test Loss: {:.4f}, accuracy: {:.2f}%\n".format(epoch, test_loss, test_accuracy))
```
    
    Train Epoch: 10 [0/397 (0%)]	Loss: 0.636741
    Train Epoch: 10 [80/397 (20%)]	Loss: 0.449106
    Train Epoch: 10 [160/397 (40%)]	Loss: 0.607533
    Train Epoch: 10 [240/397 (60%)]	Loss: 0.713035
    Train Epoch: 10 [320/397 (80%)]	Loss: 0.705570
    [10] Test Loss: 0.6319, accuracy: 66.25%
    


## 4. 내장 모델 사용하기(Transfer Learning 사용)
- pretrained된 weight를 사용하면, 마지막 Feature는 1*1000(ImageNet의 Class 갯수)이다.
- 따라서 마지막에 출력되는 Feature의 갯수를 지금 나의 데이터의 Class의 갯수로 맞춰줘야 한다. 따라서 다음과 같은 작업을 수행한다.    

```python
model = models.resnet18(pretrained = True) 
num_ftrs = model.fc.in_features   # fully connected layer에 들어가기 직전의 feature의 갯수(numbers)를 알아온다. 
model.fc = nn.Linear(num_ftrs, 2) 
# model의 마지막 fully connected layer의 정의를 바꿔 버린다.  
model.fc = nn.Linear(num_ftrs, 2) 
# 원래는 num_ftrs -> 1000 이었다면, num_ftrs -> 2 으로 바꾼다. 


if USE_CUDA:
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr = 0.0001)
EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    train(model, dataloaders["train"], optimizer, epoch)
    test_loss, test_accuracy = evaluate(model, dataloaders["val"])
    print("[{}] Test Loss: {:.4f}, accuracy: {:.2f}%\n".format(epoch, test_loss, test_accuracy))
```

    Train Epoch: 10 [0/397 (0%)]	Loss: 0.821322
    Train Epoch: 10 [80/397 (20%)]	Loss: 0.526726
    Train Epoch: 10 [160/397 (40%)]	Loss: 0.820258
    Train Epoch: 10 [240/397 (60%)]	Loss: 0.242522
    Train Epoch: 10 [320/397 (80%)]	Loss: 0.604658
    [10] Test Loss: 0.2158, accuracy: 93.70%
    

- param.requires_grad = False : Backward에 의해서 gradient 계산이 안된다.
- 아래와 같이 코딩을 하면 'nn.Linear(num_ftrs, 2)'로 정의한 마지막 Layer만 Backpropagation으로 weight가 update되지만, 나머지 pretrained weight 즉 가져온 weight는 변하지 않는다.    

```python
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False 
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr = 0.0001)

EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    train(model, dataloaders["train"], optimizer, epoch)
    test_loss, test_accuracy = evaluate(model, dataloaders["val"])
    print("[{}] Test Loss: {:.4f}, accuracy: {:.2f}%\n".format(epoch, test_loss, test_accuracy))
```

    
    Train Epoch: 10 [0/397 (0%)]	Loss: 0.773592
    Train Epoch: 10 [80/397 (20%)]	Loss: 0.575552
    Train Epoch: 10 [160/397 (40%)]	Loss: 0.498209
    Train Epoch: 10 [240/397 (60%)]	Loss: 0.761115
    Train Epoch: 10 [320/397 (80%)]	Loss: 0.598199
    [10] Test Loss: 0.6826, accuracy: 59.45%
    


