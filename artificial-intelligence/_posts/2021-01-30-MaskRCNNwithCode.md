---
layout: post
title: 【In-Segmen】Understanding Mask-RCNN(+RPN) paper with code 
---

- **논문** : [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)
- **분류** : Original Instance Segmentation
- **저자** : Kaiming He, Georgia Gkioxari (Facebook AI Research)
- **읽는 배경** : Recognition Basic. Understand confusing and ambiguous things.
- **읽으면서 생각할 포인트** : 코드와 함께 최대한 완벽히 이해하기. 이해한 것 정확히 기록해두기.
- **느낀점**  : 
  1. RoIAlign 논문 보면 이해할 수 있게 만들어 놓은 줄 알았는데, 그렇지도 않다. 차라리 아래의 Post글을 보는게 훨씬 좋다. 이런걸 보면, **논문을 읽고, 나의 생각을 조금 추가해서 이해하는게 정말 필요한듯 하다.** 논문에는 정확한 설명을 적어놓은게 아니므로. 
  2. 논문 요약본 보다는, 직관적(intuitive) 이해를 적어놓은 유투브나, 아래의 Bilinear interpolation과 같은 블로그를 공부하는게 자세한 이해, 완벽한 이해를 가능케 하는 것 같다.
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

1. Abstract, Introduction, Related Work
2. Mask R-CNN
   - Mask Representation :
     1. mask-branch output : \[m^2 x K channel\] binary(sigmoid) mask (K-classes, mxm masks of resolution)
     2. fc layer를 사용하는 것보다, FCN 개념의 convolutions을 사용함으로써 더 좋은 결과. spatial dimensions information을 읽지 않을 수 있었다. 
   - but Loss_mask is only defined on the k-th mask(K channel 중 k번째 채널)
   - RoIAlign :  bilinear interpolation, ROI를 n x n으로 자른 후의 한 cell을 논문에서는 bin이라고 표현함. ROIAlign은 논문에서 이해할 수 없게 적어놓았다. 
   - Network Architecture :  straightforward structure bask-branch
   - RPN 개념은 faster-rcnn을 그대로 이용했으므로, mask-rcnn 코드를 보고 RPN의 활용을 좀 더 구체적으로 공부해 보자. faster-rcnn 논문 보지말고 이전 나의 블로그 포스트 참조([20-08-15-FastRCNN](https://junha1125.github.io/blog/artificial-intelligence/2020-08-15-1FastRCNN/))



# 2. Detectron2 - MaskRCNN

1. Detectron2 전반적인 지식은 다음 게시물 참조 (210131-Detectron2 Tutorial and Overview)



# 3. multimodallearning/pytorch-mask-rcnn

1. Github Link : [multimodallearning/pytorch-mask-rcnn](multimodallearning/pytorch-mask-rcnn)
2. 원래 이해가 안됐다가, 이해가 된 **[RPN] [ROI-Align] [Mask-Branch] [Loss_mask]** 에 대해서 코드로 공부해보자. 