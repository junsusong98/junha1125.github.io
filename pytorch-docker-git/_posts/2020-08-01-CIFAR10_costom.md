---
layout: post
title: 【Pytorch 실습】 CIFAR10 데이터셋. Dropout, Xivier Weight + ResNet18 사용하기
description: > 
        아래의 코드는 Kaggle 및 Git의 공개된 코드를 적극 활용한, 과거의 공부한 내용을 정리한 내용입니다.  
---
【Pytorch 실습】 CIFAR10 데이터셋. Dropout, Xivier Weight + ResNet18 사용하기
# 1. Dropout 사용해보기  

## 1. 데이터 Load
```python
import torch
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
```

- 공개적인 데이터를 다음과 같이 datasets.~~ 해보면 정말 많은 데이터가 있는 것을 확인할 수 있다.  
- Trainsforms.compose를 이용해서 Tensor형으로 바꿔주는 전처리를 해줄 수 있다. 
- Normalize(평균, 분산) -> RGV에 대해서 정규처리를 해줄 수 있다. 
```python
from torchvision import transforms, datasets
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("../data/CIFAR_10/",
                     train = True,
                     download = True,
                     transform = transforms.Compose([
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5),
                                              (0.5, 0.5, 0.5))])), batch_size = 64, shuffle = True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("../data/CIFAR_10",
                     train = False,
                     transform = transforms.Compose([
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5),
                                              (0.5, 0.5, 0.5))])), batch_size = 64 )
```

## 2. 모델 구현
- 드롭아웃을 하는 방법은 아래와 같다. 여기서는 Fully Connected layer에서 dropout을 사용했는데, 내부 동작원리는 간단하다. 
- layer의 node 중 랜덤하게 일정비율로 0으로 만들어 버리는 것이다. (node가 동작하지 않는다.) 이렇게 함으로써 간단하게 dropout을 구현할 수 있다. 
- 아래에서 사용한 변수 self.dropout_o = 0.2가 랜덤하게 노드를 잠그는 비율이다. 

```python
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(2 * 2 * 64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4_bn = nn.BatchNorm2d(64)
        
        self.dropout_p = 0.2  # dropout 확률 
        
    def forward(self, x):
        x = self.conv1(x) # 32 * 32 * 3 -> 32 * 32 * 8
        x = self.conv1_bn(x)
        x = F.tanh(x)     # 32 * 32 * 8
        x = self.pool(x)  # 16 * 16 * 8
        
        x = self.conv2(x) # 16 * 16 * 8 -> 16 * 16 * 16
        x = self.conv2_bn(x)
        x = F.tanh(x)     # 16 * 16 * 16
        x = self.pool(x)  # 8  *  8 * 16
        
        x = self.conv3(x) # 8 * 8 * 16 -> 8 * 8 * 32
        x = self.conv3_bn(x)
        x = F.tanh(x)     # 8 * 8 * 32
        x = self.pool(x)  # 4 * 4 * 32
        
        x = self.conv4(x) # 4 * 4 * 32 -> 4 * 4 * 64
        x = self.conv4_bn(x)
        x = F.tanh(x)     # 4 * 4 * 64
        x = self.pool(x)  # 2 * 2 * 64
        
        x = x.view(-1, 2 * 2 * 64)
        x = self.fc1(x)
        x = F.dropout(x, p = self.dropout_p)  # actiavtion function 전후 상관 없다. 알아서..
        x = F.relu(x)
        x = self.fc2(x)
        x = F.dropout(x, p = self.dropout_p)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim = 1)
        return x

print("DEVICE: ", DEVICE)
print("MODEL: ", model)

```

    DEVICE:  cuda
    MODEL:  CNN(
      (conv1): Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv4): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (fc1): Linear(in_features=256, out_features=64, bias=True)
      (fc2): Linear(in_features=64, out_features=32, bias=True)
      (fc3): Linear(in_features=32, out_features=10, bias=True)
      (conv1_bn): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2_bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3_bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv4_bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )


## 3. 가중치 초기화 하기
- 보통은 Transfer Learning을 통해서 가중치 초기화에 대해서 공부할 기회가 없지만 여기서는 가중치 초기화를 직접 해보았다.  
- 가중치를 초기화 하는 방법에는 Xavier와 He등이 있다. 여기서는 Xavier를 사용해 보았다. Relu를 사용한 모델에서는 He를 사용하는게 더 유리하다고 하다.  
```python

import torch.nn.init as init
def weight_init(m):
    
    '''
    이 주석을 풀면 Xavier, kaiming 에 대한 Initialization에 대한 상세 설정을 할 수 있다. 

    Ref: https://pytorch.org/docs/stable/nn.init.html
    
    init.uniform_(tensor, a = 0.0, b = 1.0) (a: Lower bound, b: Upper bound)
    init.normal_(tensor, mean = 0.0, std = 1.0)
    init.xavier_uniform_(tensor, gain = 1.0)
    init.xavier_normal_(tensor, gain = 1.0)
    init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
    init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
    '''
    
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)

    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        init.normal_(m.bias.data)
        
model = CNN().to(DEVICE)
model.apply(weight_init)

optimizer = optim.Adam(model.parameters(), lr = 0.001)
print("DEVICE: ", DEVICE)
print("MODEL: ", model)
```

## 4. 모델 학습 및 추론
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
        
        if batch_idx % 100 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, 
                batch_idx * len(data), 
                len(train_loader.dataset), 
                100. * batch_idx / len(train_loader), 
                loss.item()))
```


```python
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


```python
EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, epoch)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("[{}] Test Loss: {:.4f}, accuracy: {:.2f}%\n".format(epoch, test_loss, test_accuracy))
```

 
    Train Epoch: 10 [0/50000 (0%)]	Loss: 1.137886
    Train Epoch: 10 [6400/50000 (13%)]	Loss: 0.988332
    Train Epoch: 10 [12800/50000 (26%)]	Loss: 0.784159
    Train Epoch: 10 [19200/50000 (38%)]	Loss: 1.073874
    Train Epoch: 10 [25600/50000 (51%)]	Loss: 1.101201
    Train Epoch: 10 [32000/50000 (64%)]	Loss: 1.111024
    Train Epoch: 10 [38400/50000 (77%)]	Loss: 0.956936
    Train Epoch: 10 [44800/50000 (90%)]	Loss: 0.914141
    [10] Test Loss: 1.0315, accuracy: 64.86%
    



```python
print("CNN's number of Parameters: ", sum([p.numel() for p in model.parameters()]))
```

    CNN's number of Parameters:  43626




---

# 2. ResNet 사용해보기  
## 1. 데이터 Load  
  
```python
import torch
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
```


```python
from torchvision import transforms, datasets
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("../data/CIFAR_10/",
                     train = True,
                     download = True,
                     transform = transforms.Compose([
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5),
                                              (0.5, 0.5, 0.5))])), batch_size = 64, shuffle = True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("../data/CIFAR_10",
                     train = False,
                     transform = transforms.Compose([
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5),
                                              (0.5, 0.5, 0.5))])), batch_size = 64 , shuffle = True)
```

## 2. ResNet 모델 구현하기

```python
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BasicBlock(nn.Module): # bottlenexk 이라는 용어로도 많이 사용된다.
    def __init__(self, in_planes, planes, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(planes))
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module): # _ㄴmake_layer라는 용어로 많이 사용되니, 알아두기
    def __init__(self, num_classes = 10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride = 1)
        self.layer2 = self._make_layer(32, 2, stride = 2)
        self.layer3 = self._make_layer(64, 2, stride = 2)
        self.linear = nn.Linear(64, num_classes)
        
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks  - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
```


```python
model = ResNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr = 0.001)

print("DEVICE: ", DEVICE)
print("MODEL: ", model)
```
## 3. 학습 및 추론

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
        
        if batch_idx % 100 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, 
                batch_idx * len(data), 
                len(train_loader.dataset), 
                100. * batch_idx / len(train_loader), 
                loss.item()))
```


```python
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


```python
EPOCHS = 3
for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, epoch)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("[{}] Test Loss: {:.4f}, accuracy: {:.2f}%\n".format(epoch, test_loss, test_accuracy))
```
    
    Train Epoch: 3 [0/50000 (0%)]	Loss: 1.127532
    Train Epoch: 3 [6400/50000 (13%)]	Loss: 1.081882
    Train Epoch: 3 [12800/50000 (26%)]	Loss: 0.691810
    Train Epoch: 3 [19200/50000 (38%)]	Loss: 0.759505
    Train Epoch: 3 [25600/50000 (51%)]	Loss: 0.786468
    Train Epoch: 3 [32000/50000 (64%)]	Loss: 0.904780
    Train Epoch: 3 [38400/50000 (77%)]	Loss: 0.772286
    Train Epoch: 3 [44800/50000 (90%)]	Loss: 0.593489
    [3] Test Loss: 0.7578, accuracy: 73.14%
    
# 3. 유명한 모델 그대로 가져와 사용하는 방법
- 기본적으로 Pytorch에 저장되어 있는 모델 사용하기.
- 아래와 같이 굳이 모델을 위에 처럼 구현하지 않아도 아래처럼 쉽게, 학습시킬 수 있다. 

```python
import torchvision.models as models
model = models.resnet18().cuda()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
EPOCHS = 3
for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, epoch)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("[{}] Test Loss: {:.4f}, accuracy: {:.2f}%\n".format(epoch, test_loss, test_accuracy))
```
    
    Train Epoch: 3 [0/50000 (0%)]	Loss: 0.853458
    Train Epoch: 3 [6400/50000 (13%)]	Loss: 0.816588
    Train Epoch: 3 [12800/50000 (26%)]	Loss: 0.678977
    Train Epoch: 3 [19200/50000 (38%)]	Loss: 0.873804
    Train Epoch: 3 [25600/50000 (51%)]	Loss: 0.783468
    Train Epoch: 3 [32000/50000 (64%)]	Loss: 1.123129
    Train Epoch: 3 [38400/50000 (77%)]	Loss: 0.890110
    Train Epoch: 3 [44800/50000 (90%)]	Loss: 0.780257
    [3] Test Loss: 0.8566, accuracy: 70.31%
    
--- 


```python
resnet18 = models.resnet18()
alexnet = models.alexnet()
vgg16 = models.vgg16()
squeezenet = models.squeezenet1_0()
densenet = models.densenet161()
inception = models.inception_v3()
googlenet = models.googlenet()
shufflenet = models.shufflenet_v2_x1_0()
mobilenet = models.mobilenet_v2()
resnext50_32x4d = models.resnext50_32x4d()
wide_resnet50_2 = models.wide_resnet50_2()
mnasnet = models.mnasnet1_0()
```
