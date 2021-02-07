---
layout: post
title: 【LightWeight】Understanding DenseNet, MobileNet V3 from youtube w/ code
---

- **논문1** : [MobileNet V1 paper link](https://arxiv.org/abs/1704.04861)  / [Youtube Link1](https://www.youtube.com/watch?v=7UoOFKcyIvM)
- **논문2** : [MobileNet V2 paper link](https://arxiv.org/abs/1801.04381)  / [Youtube Link2](https://www.youtube.com/watch?v=mT5Y-Zumbbw)
- **논문2** : [Xception YoutubeLink](https://www.youtube.com/watch?v=V0dLhyg5_Dw&t=10s)
- **분류** : Object Detection
- **공부 배경** : MobileNetV3 Paper 읽기 전 선행 공부로 youtube 발표 자료 보기
- **목자**
  4. (이전 Post) Inception Xception GoogleNet ([바로가기](https://junha1125.github.io/blog/artificial-intelligence/2021-02-06-MobileNet_youtube/#1-inception-xception-googlenet))
  2. (이전 Post) MobileNet v1 ([바로가기](https://junha1125.github.io/blog/artificial-intelligence/2021-02-06-MobileNet_youtube/#2-mobilenetv1))
  3. (이전 Post) MobileNet v2 ([바로가기](https://junha1125.github.io/blog/artificial-intelligence/2021-02-06-MobileNet_youtube/#3-mobilenetv2))
  4. MobileNet V2 - Manifold of Interest , Linear Bottlenecks Code (바로가기)
  5. MobileNet V3 (바로가기)
  6. DenseNet (바로가기)





# 1. Linear Bottlenecks Code

- 결국 Relu안하는 Convolution Layer이다. ㅅ... 욕이 나온네. "별것도 아닌거 말만 삐까번쩍하게 한다. **이게 정말 수학적으로 증명을 해서 뭘 알고 논문에서 어렵게 설명하는건지?**, 혹은 **그냥 그럴 것 같아서 실험을 몇번 해보니까 이게 성능이 좋아서 멋진 말로 끼워맞춘건지 모르겠다**.

- [Github Link](https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py)

  1. MobileNet V2 코드 - 파일이 1개 뿐이고 생각보다 엄청 쉽다. 이건 뭐 FPN 파일 하나 코드보다 훨씬 이해하기 쉽다.

  2. ```python
     class InvertedResidual(nn.Module):
         def __init__(self, inp, oup, stride, expand_ratio):
             super(InvertedResidual, self).__init__()
             self.stride = stride
             assert stride in [1, 2]
     
             hidden_dim = int(inp * expand_ratio)
             self.use_res_connect = self.stride == 1 and inp == oup
     
             if expand_ratio == 1:
                 self.conv = nn.Sequential(
                     # dw
                     nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                     nn.BatchNorm2d(hidden_dim),
                     nn.ReLU6(inplace=True),
                     # pw-linear ***** 이거가 Linear Bottlenecks *****
                     nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                     nn.BatchNorm2d(oup),
                 )
             else:
                 self.conv = nn.Sequential(
                     # pw
                     nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                     nn.BatchNorm2d(hidden_dim),
                     nn.ReLU6(inplace=True),
                     # dw
                     nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                     nn.BatchNorm2d(hidden_dim),
                     nn.ReLU6(inplace=True),
                     # pw-linear ***** 이거가 Linear Bottlenecks *****
                     nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                     nn.BatchNorm2d(oup),
                 )
     ```

- [Manifold 학습 이란](https://markov.tistory.com/39) 

  1. 매니폴드 학습이란, <그림 1>과 같은 형태로 수집된 정보로부터 <그림 2>와 같이 **바르고 곧은** 유클리디안 공간을 찾아내는 것을 말합니다.
  2. 아래의 그림처럼, 100차원 vector들이 100차원 공간상에 놓여있다. 하지만 유심히 보면, 결국 백터들은 2차원 평면(왼쪽 그림) 혹은 3차원 공간(오른쪼 그림) 으로 이동 시킬 수 있다. 즉 100차원 공간이라고 보여도 사실은 2차원 3차원으로 더 잘 표현할 수 있는 백터**들** 이었다는 것이다.
  3. 이렇게 높은 차원의 백터**들**을, 낮은 차원의 직관적이고 곧은 공간으로 이동시킬 수 있게 신경망과 신경망의 파라메터를 이용한다. 
  4. <img src="C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210207134456533.png" alt="image-20210207134456533" style="zoom:80%;" />

- [Manifold 란?](https://greatjoy.tistory.com/51) 

  1. ![image-20210207134724874](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210207134724874.png)
  2. 스위스롤(왼쪽그림)을 우리가 보면 3차원이지만, 그 위에 올라가 있는 개미가 느끼기엔 2차원이다. 
  3. 매니폴드 학습이란 '눈을 감고(학습이되지 않은 상태에서) 비유클리디안 형태를 만져가며(데이터를 이용해) 모양을 이해해(모델을 학습해) 나가는 것이다'. 즉, 스위스롤을 2차원 평면으로 피는 과정이다. 
  4. 오른쪽 그림과 같이 2차원 데이터를, 1차원으로 필 수 있다면 데이터와의 유사도 측정등에 매우 좋을 것이다. 
  5. 혹은 차원을 늘리기도 한다. 예를 들어 오른쪽 그림의 실을 잡고 들어올리는 행위가, 차원을 늘리는 행위이고, 그것은 신경망의 Layer를 추가하는 행위라고도 할 수 있다. 

- [Manifold of Interest](https://stats.stackexchange.com/questions/465208/meaning-of-manifold-of-interest) 위의 내용까지 간략히 요약 정리!

  1. 실용적으로 말해서! "**conv혹은 fc를 통과하고 activation function을 통과하고 나온 (Subset) Feature가 Manifold 이다!**" 그리고 "**그 Manifold 중에 우리가 관심있어하는 부분, 가장 많은 이미지의 정보를 representation하는 일부를 논문에서는 Manifold of interest 라고 한다**."

- [**Linear Bottlenecks**](http://www.navisphere.net/6145/mobilenetv2-inverted-residuals-and-linear-bottlenecks/)

  1. ![image-20210207135326788](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210207135326788.png)
  2. 논문 설명 + 나의 이해 : Input data = x 가 맨 왼쪽과 같이 표현 된다고 하자. 그리고 Activation_Function( T \* x) 를 수행한다. 여기서 T는 Input data의 차원을 위 사진의 숫자와 같이 2,3,5,15,30 차원으로 늘려주는 FC Layer라고 생각하자. 
  3. 결과를 살펴보면, 낮은 차원으로 Enbeding하는 작업은, 특히 (비선형성 발생하게 만드는) Relu에 의해서 (low-dimensional subspace = 윗 이미지 그림을 살펴 보면) 정보의 손실이 많이 일어난다. 하지만 고차원으로의 Enbeding은 Relu에 의해서도 정보의 손실이 그리 많이 일어나지 않는다.
  4. **솔직히 논문이나 PPT내용이 무슨소리 하는지는 모르겠다.** 
  5. **결국에는 위와 같은 "정보 손실"을 줄이기 위해서, 비선형이 Relu를 쓰지 않고 Linear Function을 쓰겠다는 것이다.** 



\<아래 이미지가 안보이면 이미지가 로딩중 입니다\>     



# 2. MobileNet V3





# 3. DenseNet 



