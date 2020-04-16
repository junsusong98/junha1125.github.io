---
layout: post
title: (논문 코드) feature pyramid networks - Code review
# description: > 
    
---
(논문리뷰) feature pyramid networks - Code review

# 코드를 설명하는 그림

![picture1](https://user-images.githubusercontent.com/46951365/77819391-da60f900-711d-11ea-95a4-68e02f0f676a.jpg)

# 코드

```python
'''
FPN in PyTorch.
See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


# 4. block == Bottlenect (for Bottom-up)
class Bottleneck(nn.Module):
    expansion = 4

    # layer1 : (64, 256), (64, 64) , (1 , 1) // layer2 : (256, 512), (128, 128) , (2, 1) // layer3 : (512, 1024), (256, 256) , (2, 1)
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)  
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        # 5. shortcut 정의 하기 = Resnet 을 한다.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes: # layer1,2,3,4의 for 2개에서 모두 shortcut이 이루어진다. 
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # x's channel-> layer1 : 64 // layer2 : 128 // layer3 : 256  // layer4 : 512
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))  # out's channel-> layer1 : 256 // layer2 : 512 // layer3 : 1024  // layer4 : 2048
        out += self.shortcut(x)          # bottom-up에서 Resnet의 구조가 작동하는 부분. x와 out의 channel이 다른것은 어떻게 처리하는지는 위에 정의 되어 있다.
        out = F.relu(out)
        return out



class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        # 1. input channel 3 (RGB) // output channel 64 // kernel size 7*7 // stride가 1이면 같은 size output이지만, stride = 2여서 output == input//2 size
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn1 = nn.BatchNorm2d(64) 


        # 2. Bottom-up layers. 4개의 layer를 통과하고 나오는 결과 : channel 2048, width와 high는 대략 /(4(#1)*8(layer2,3,4)) 가 된다.. 하나의 _make_layer에 의해서 2로 나눠지므로.
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

    # 3. call block (for Bottom-up)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1) # [1or2] + [1] = [1or2 , 1] 
        layers = []
        for stride in strides:
            # layer1 : (64, 256), (64, 64) , (1 , 1) // layer2 : (256, 512), (128, 128) , (2, 1) // layer3 : (512, 1024), (256, 256) , (2, 1) // in_plain, planes, sride
            layers.append(block(self.in_planes, planes, stride))  
            self.in_planes = planes * block.expansion            
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y
        # upsampling mode 설명 : https://pytorch.org/docs/0.3.1/_modules/torch/nn/modules/upsampling.html#Upsample

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1) # 1. 위의 7*7 conv와 max_pool에 의해서 w/h가 /4 처리가 된다. 
        c2 = self.layer1(c1)  # c2 size = torch.Size([1, 256, 150, 225])
        c3 = self.layer2(c2)  # c3 size = torch.Size([1, 512, 75, 113])
        c4 = self.layer3(c3)  # c4 size = torch.Size([1, 1024, 38, 57])
        c5 = self.layer4(c4)  # c5 size = torch.Size([1, 2048, 19, 29])
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5


def FPN101():
    # return FPN(Bottleneck, [2,4,23,3])
    return FPN(Bottleneck, [2,2,2,2])


def test():
    net = FPN101()
    fms = net(Variable( torch.randn(1,3,600,900) )) # from torch.autograd import Variable -> dL/dw 처럼 w에 관하여 미분한 값을 구할때 사용. 여기서 randn이 w이다.
    for fm in fms:
        print(fm.size())

test()

"""
input : torch.size([1, 3, 600, 900])

fms[0] = torch.Size([1, 256, 150, 225])   
fms[1] = torch.Size([1, 256, 75, 113])
fms[2] = torch.Size([1, 256, 38, 57])
fms[3] = torch.Size([1, 256, 19, 29])

"""
```

