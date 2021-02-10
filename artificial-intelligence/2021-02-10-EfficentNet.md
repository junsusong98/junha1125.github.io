---
layout: post
title: 【Detection】Understanding EfficentNet+EfficentDet paper w/ code
---

- **논문** : [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf)

- **분류** : Object Detection

- **저자** : Mingxing Tan 1 Quoc V. Le 1

- **읽는 배경** : Recognition Basic. Understand confusing and ambiguous things.

- **읽으면서 생각할 포인트** : 코드와 함께 최대한 완벽히 이해하기. 이해한 것 정확히 기록해두기.

- **느낀점**  : 

  - ㅇ

- **목차**
  
  1. EfficientNet from youtube ([바로가기](https://junha1125.github.io/blog/artificial-intelligence/2021-02-10-EfficentNet/#1-efficientnet-from-youtube))
  2. EfficientNet Paper ([바로가기](https://junha1125.github.io/blog/artificial-intelligence/2021-02-10-EfficentNet/#2-efficientnet-paper))
  3. Code - lukemelas/EfficientNet-PyTorch ([바로가기](https://junha1125.github.io/blog/artificial-intelligence/2021-02-10-EfficentNet/#3-lukemelasefficientnet-pytorch))
  4. EfficientDet from youtube (바로가기)
  5. EfficientDet Paper (바로가기)
  6. Code - zylo117/Yet-Another-EfficientDet-Pytorch (바로가기)
  





# 1. EfficientNet from youtube

- [youtube 논문 발표 링크](https://www.youtube.com/watch?v=11jDC8uZL0E&t=104s) 
- 확실하지는 않지만, EfficientNet은 MobileNetV3의 다음 버전 같다. 하지만 사람들은 각각의 장점이 있다고 한다. 다양한 조건에서 고려하면 뭐가 더 좋고 나쁘다 라고 아직은 할 수 없다. 
- 논문의 결과에서도 MobileNet과의 비교는 하지 않았다. NasNet등과 비교를 했을 뿐이다.





# 2. EfficientNet Paper





# 3. lukemelas/[EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)

## 3-1 Readme

- EfficientNet은 a family of image **classification** models 이다. 

- Based on **MnasNet** in term of **AutoML**, **Compound Scaling**, 

- EfficientNet-B0가 Mobile-size의 Baseline Network이고, 그 후 scale up을 한 EfficientNet-B1 to B7 을 만들어 낸다.

- 아래의 그래프를 통해서, ResNet50, ResNet152, DensNet, NASnet등 과의 속도 및 성능 비교를 해보자 : 내 생각에는 Time efficiency면서, ResNet50보다 Parameter수가 많은 것은 아무리 정확도가 높아도 큰 의미 없는 것 같다. 따라서 B4까지만 고려를 한다면, DenseNet과 비교해서도 훨씬 좋은 정확도를 가지고 있다.    
  ![image-20210209181603689](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210209181603689.png)

- MobileNetV3 VS EfficientNet을 비교하는 글에서, EfficientNet이 Transfer laerning에서 그리 안 좋다고 한다. (왜지?) 뭐가 더 좋다고 말 할 수는 없는 것 같다.

- EfficientNet이 Google에서 만들어졌기 때문에, Tensorflow기반이고, 이 코드는 Pytorch로 re-implementation된 코드라고 한다. [Original code](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)는 여기 있다. weight를 reload해서 Pytorch에 넣기 위해 노력했고 Pytorch 모델 또한 simple, flexible(유연,유동성), and extensible(확장가능성)에서 좋은 결과 얻기 위해 노력했다.

- 내가 이 코드을 이용하기 위한 과정은 아래와 같이 심플하다.

  - ```python
    import json
    from PIL import Image
    import torch
    from torchvision import transforms
    
    from efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b0')
    
    # 0. Preprocess image
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    img = tfms(Image.open('img.jpg')).unsqueeze(0)
    print(img.shape) # torch.Size([1, 3, 224, 224])
    
    # 0. Load ImageNet class names
    labels_map = json.load(open('labels_map.txt'))  # 이미 EfficientNet-Pytorch/examples/simple/labels_map.txt 있다.
    labels_map = [labels_map[str(i)] for i in range(1000)]
    
    # 1. For Classify
    model.eval()
    with torch.no_grad():
        outputs = model(img)
    
    # 2. For Feature Extractor
    features = model.extract_features(img)
    print(features.shape) # torch.Size([1, 1280, 7, 7])
    ```





# 4. EfficientDet from youtube



# 5. EfficientDet Paper



# 6. zylo117/**[Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)**