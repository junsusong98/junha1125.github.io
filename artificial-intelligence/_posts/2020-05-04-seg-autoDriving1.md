---
3layout: post
title: 【Paper】 A Brief Survey and an Application of Semantic Image Segmentation for Autonomous Driving
description: >  
    Image Segmentation for Autonomous Driving 논문 리뷰 및 정리
---

(위성Segment) Segmentation for Autonomous Driving 논문 정리
논문 원본 : **2019 (cite:10) [A Brief Survey and an Application of Semantic Image Segmentation for Autonomous Driving](https://arxiv.org/abs/1808.08413)** 


## 1. [논문 필기 및 정리 자료 다운로드](https://github.com/junha1125/Imgaes_For_GitBlog/tree/master/2020-05-04)

<br>

## 2. 논문 전체 요약
- 이 논문에서는 deep learning이 무엇이고 CNN이 무엇인지에 대한 아주 기초적인 내용들이 들어가 있다. 이미 알고 있는 내용들이라면 매우 쉬운 논문이라고 할 수 있다.   
- FCN에 대한 소개를 해주고, a virtual city Image인 SYNTHIA-Rand-CVPR16 Dataset을 사용해,  FCN-AlexNet, FCN-8s, FCN-16s and FCN-32s 이렇게 4가지 모델에 적용해 본다.
- 결론 : 
   1. Maximum validation accuracies of 92.4628%, 96.015%, 95.4111% and 94.2595%
are achieved with FCN-AlexNet, FCN-8s, FCN-16 and FCN-32s models 
   2. Training times는 FCN-AlexNet이 다른 모델들에 비해 1/4시간이 걸릴 만큼 학습시간이 매우 짧다. 하지만 우리에게 중요한 것은 inference시간이므로, the most suitable
model for the application is FCN-8s.
   3. 각 모델들에 대한 inference 시간 비교는 논문에서 해주지 않는다.

## 3. 앞으로 추가 공부 계획   
- 필수 공부 : [https://paperswithcode.com/sota](https://paperswithcode.com/sota)
- 사이트 참고해서 논문 읽고 계보 파악하며 매일매일 노력할 예정!
