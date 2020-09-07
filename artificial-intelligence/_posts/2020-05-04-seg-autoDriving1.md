---
3layout: post
title: 【Paper】 Semantic Segmentation for AutoDriving/ 공부 계획 및 모델 핵심 정리
description: >  
    Image Segmentation for Autonomous Driving 논문 리뷰 및 정리
---

(위성Segment) Segmentation for Autonomous Driving 논문 정리
논문 원본 : **2019 (cite:10) [A Brief Survey and an Application of Semantic Image Segmentation for Autonomous Driving](https://arxiv.org/abs/1808.08413)** 


## 1. [논문 필기 및 정리 자료 다운로드 링크](https://github.com/junha1125/Imgaes_For_GitBlog/tree/master/2020-05-04)

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
- 사이트 참고해서 논문 읽고 계보 파악하며 매일매일 공부할 예정!

## 4. 면접 준비 - 딥러닝 모델 핵심 정리

각 모델별 핵심 용어 정리(논문 level2 정도로만 읽은 상태. 빠르게 Skip. 핵심만 파악)
1. 코드는 직접만드는 것보다는, 이미 만들어진 코드들을 수정하고 참고
   - FPN : bottomup[bottleNet+resnet], topdown[ upsample, leteral connection(sum)]
   - ResNet : self.shortcut = nn.sequential(); out += self.shortcut(input)
   - vgg : 신경망은 매우 간단. data, criterion, optimizer. zero_grad -> loss -> backword -> opti.step
2. DQN – Q learning, Fixed Target Network, Experience Replay memory
3. FPN – Bottom-up, Top-down, lateral-connection, multi-scale-detection
4. Fast-RCNN – Selective search, ROI pooling, Bounding Box Regression
5. Faster-RCNN – RPN(anchor개념을 사용해서 각 grid마다 객체의 유무/ Box location 정보를 뽑아낸다. RPN의 Loss function도 따로 있다. 객체유무/ Box coordinate 차이), 9개 anchor.
6. FCN – FCN32 ->(upsampling) FCN16
7. Mask-RCNN – Align pooling(각 픽셀 4등분+weighted sum), Binary Cross Entropy, 마지막단 FCN, Multi training(keypoint Detection), 최초 Instacne Segmentation
8. DeepLab – Dilated Con(global fearture/contexts), CRP(Posterior, Energy), ASPP(atrous spatial pyramid pooling)
9. PSPNet – Pyramid Pooing layer, concatenate
10. ParsNet – Global context information, Global pooling, 
11. SegNet – Encoder/Decoder, maxpooling index
12. HRNet – high resolution, low resolution의 information Exchange. BackboneNetwork
13. Panotic-Network (PA-Net) – FPN + mask+rcnn
14. Dilated Conv – 추가 비용없이, receptive field 확장
15. RNN, LSTM, Attention, GAN 개념 사용 가능
16. CRF – Posterior를 최대화하고 Energy를 최소화한다.  
   Energy는 위치가 비슷하고 RGB가 비슷한 노드사이에서 라벨이 서로 다른것이라고 하면 Panalty를 부과함으로써 객체의 Boundary를 더 정확하게
찾으려고 하는 노력입니다.   
Energy공식에 비용이 너무 많이 들어서 하는 작업이 mean field approximation이라고 합니다. 
17. Yolo – 1stage detector, cheaper grid, 7*7*30(5+5+20), confidence낮음버림, NMS -> Class 통합
18. SSD – Anchor, 다양한 크기 객체, 작은 -> 큰 물체, Detector&classifier(3개 Anchor, locallization(x,y,w,h), class-softMax결과(20+1(배경)) ), NMS

