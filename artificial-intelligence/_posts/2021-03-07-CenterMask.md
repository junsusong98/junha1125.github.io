---
layout: post
title: 【In-Segmen】CenterMask : Real-Time Anchor-Free Instance Segmentation
---

- **논문** : [CenterMask : Real-Time Anchor-Free Instance Segmentation](https://arxiv.org/abs/1911.06667)
- **분류** : Real Time Instance Segmentation
- **저자** : Youngwan Lee, Jongyoul Park
- **느낀점 :** 
  - 
- **목차**
  1. Paper Review
  2. Code Review
     - Github link : [https://github.com/tianzhi0549/FCOS](https://github.com/tianzhi0549/FCOS)



# CenterMask

# 1. Conclusion, Abstract, Introduction

- Real-time anchor-free one-stage
- VoVNetV2 backbone : (1) residual connection (2) effective Squeeze-Excitation (eSE, Squeeze and Excitation Block (M2Det, MobileNetV3, EfficientDet 참조))
- Spatial attention guided mask (=SAG-Mask) : Segmentation Mask를 예측한다. spatial attention을 사용함으로써 집중해야할 Informative Pixel에 집중하고, Noise를 Suppress한다. 
- ResNet-101-FPN backbone / 35fps on Titan Xp
- PS. Introduction은 읽을 필요는 것 같고, Relative Work는 아에 없다. Introduction에 Relative Work의 내용들이 가득해서 그런 것 같다. 



# 2. CenterMask architecture

