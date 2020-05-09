---
layout: post
title: (위성Segment) Segmentation 공부할 코드 사이트 정리
description: >  
    공부할 코드를 찾아보고, 안 내용을 기록해 놓았습니다.
---
(위성Segment) Segmentation 공부할 코드 사이트 정리



# 1. 전체 Segmentation Models 정리 GIT

1. [semantic-segmentation-pytorch ](https://github.com/CSAILVision/semantic-segmentation-pytorch)  

   (3.2 star) 난이도 4.5 / PPM PSP Uper HRNet 같은 최신 Segmentation 코드

2. **[pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg)**  

   (2.6 star) 난이도 3 / FCN PSP U-Net 같은 기초+좋은 코드와 DataLoaders가 있다. Config 만드는 것이 필수

3. [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)   

   (1.6 star) 난이도 5 / pip install로 Segmentation을 수행할 수 있다. UNet PSP DeepLabv3 

4. [awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)  

   (1.1 star) 난이도 2 / Segmentation 모델들 정말 많다. But 대부분 중요한것 아님.. 



**따라서 나는 2번의 코드를 공부해보기로 결정했다. 하는 김에 전체 코드를 잘근잘근 씹어먹자**



# 2. Pytorch-semseg GIT 내부 내용 정리

[segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)에 구현되어 있는 코드들을 간략히 설명해주는 **[이 사이트](https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html)**를 공부한 내용을 아래에 정리해 놓는다. 

```
<전체 목차>
1. Segmentation, Pytorch란?
2. FCN
3. SegNet
4. U-Net
5. DenseNet
6. E-Net & Link-Net
7. Mask-RCNN
8. PSP-Net
9. RefineNet
10. G-FR-Net
11. DeCoupleNet
12. GAN-Approach
13. Dataset
```

***

## 1. Segmentation, Pytorch란?





