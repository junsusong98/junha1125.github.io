---
layout: post
title: 【Detection】Understanding Mask-RCNN paper with code 
---

- **논문** : [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)
- **분류** : Original Instance Segmentation
- **저자** : Kaiming He, Georgia Gkioxari (Facebook AI Research)
- **읽는 배경** : Recognition Basic. Understand confusing and ambiguous things.
- **읽으면서 생각할 포인트** : 코드와 함께 최대한 완벽히 이해하기. 이해한 것 정확히 기록해두기.
- **느낀점**  : 
- **내 블로그 관련 Post** : 
  - (1) [Mask R-CNN by DVCR](https://junha1125.github.io/blog/artificial-intelligence/2020-09-01-1mask-rcnn/) 
    1. FCN : Pixel wise Classification 
  - (2) [Mask R-CNN Youtube](https://junha1125.github.io/blog/artificial-intelligence/2020-04-13-1mask-rcnn/) 내용 정리
    1. bounding box 내부의 객체 class에 상관없이, Box 내부에 masking 정보만 따내는 역할을 하는 Mask-branch를 추가했다. 
    2. Equivariance 연산(<-> Invariant)을 수행하는 Mask-Branch는 어떻게 생겼지? Mask-branch에 가장 마지막 단에 나오는 feature map은 (ROI Align이후 크기가 w,h라 한다면..) 2 x w \* 2 x h \* **80** 이다. 여기서 80은 coco의 class별 mask 모든 예측값이다. 80개의 depth에서 loss계산은, box의 class에 대상하는 한 channel feature만을 이용한다. 나머지는 loss 계산시 무시됨. 
    3. ROI Align 이란? : Input-EachROI. output-7x7(nxn) Pooled Feature. nxn등분(첫번째 quantization)->각4등분(두번째 quantization)->[Bilinear interpolation](https://towardsdatascience.com/understanding-region-of-interest-part-2-roi-align-and-roi-warp-f795196fc193)->각4등분에 대해 Max Pooling->(nxn) Pooled Feature Map 획득.
  - (3) [FCN ](https://junha1125.github.io/blog/artificial-intelligence/2020-04-12-paper-FCN/)
    1. 해당 [이미지](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2020-04-12/fully-conv-network-for-semeantic-segmentation_15.jpg?raw=true)(필기 무시)를 확인해보면, 어디서 deconv(파라메터 학습 필요)를 사용하고, 어디서 Bilinear Interpolation이 이뤄지는지 알 수 있다. 
    2. 최종 아웃풋 image_w x image_h x 21(classes+background)



# 1. Mask-RCNN

- 
