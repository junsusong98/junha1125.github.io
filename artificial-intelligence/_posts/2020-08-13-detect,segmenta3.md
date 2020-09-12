---
layout: post
title: 【Vision】 Detection과 Segmentation 다시 정리 3 - Framework, Module, GPU
description: >
  당연하다고 생각하지만, 아직은 공부할게 많은 Detection과 Segmentation에 대한 개념을 다시 상기하고 정리해보면서 공부해볼 계획이다.
---

Detection과 Segmentation 다시 정리 3

# 1. Object Detection을 위한 다양한 모듈 

1. Keras & Tensorflow, Pytorch
    - Customization 가능 .
    - 알고리즘 별로 구현을 다르게 해줘야함.
    - Keras와 Tensorflow는 동시에 사용되고 있다. 소스 코드에 서로가 서로를 사용하고 있으니 항상 같이 import하는 것을 추천. 

2. OpenCV의 DNN 모듈
    - 간편하게 Object Detection Inference 가능.
    - 학습이 불가능.
    - CPU 위주로 동작함. GPU 사용 불가.

3. Tensorflow Object Detection API
    - 많은 Detection 알고리즘 적용가능.
    - 다루기가 어렵고 학습을 위한 절차가 너무 복잡. 
    - 다른 오픈소스 패키지에 비해, Pretrain weight가 많이 지원된다. 다양한 모델, 다양한 Backbone에 대해서.
    - Yolo는 지원하지 않고, Retinanet에 대한 지원도 약함
    - MobileNet을 backbone으로 사용해서 실시간성, 저사양 환경에서 돌아갈 수 있도록 하는 목표를 가지고 있다. 
    - Tensorflow Version2와 Version1의 충돌. 
    - Document나 Tutorial이 부족하다.
    - Research 형태의 모듈로 안정화에 의문

4. Detectron2
    - 많은 Detection 알고리즘 적용가능
    - 상대적으로 다루기가 쉽고 절차가 간단. 


# 2. 사용하면 좋은 Keras와 Tensorflow 기반의 다양한 오픈소스 패키지들

- 아래의 코드들은 대부분 Tensorflow & Keras 를 사용하고 있습니다. 

1. Yolo - [https://github.com/qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)  
    - 심플하면서 좋은 성능 보유 
    - 조금 오래 됐고, 업그레이드가 잘 안됨.

2. Retinanet - [https://github.com/fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet)  
    - 정교한 검출 성능, 작은 물체라도 잘 검출하는 능력이 있음

3. Mask R-CNN - [https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)  
    - Instance Segmentation의 중심. 
    - 다양한 기능과 편리한 사용성으로 여러 프로젝트에 사용된다. 

4. Open CV - DNN Pakage [Document](https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html)
    - 컴퓨터 비전 처리를 널리 사용되고 있는 범용 라이브러리
    - 오픈소스 기반 실시간 이미지 프로세싱에 중점을 둔 라이브러리.
    - Deep Learning 기반의 Computer Vision 모듈 역시 포팅되어 있음.
    - OpenCV가 지원하는 딥러닝 프레임워크. 이것을 사용해 Inference.
        - Caffe, Tensorflow, Torch, Darknet
    - 모델만 Load해서 Inference하면 되는 장점이 있기 때문에 매우 간편
    - CPU 위주이며, GPU 사용이 어렵다. 

# 3. GPU 활용
1. CUDA : GPU에서 병렬처리 알고리즘을 C언어를 비롯한 산업 표준 언어를 사용하여 작성할 수 있도록 하는 기술. GPU Driver Layer 바로 위에서 동작한다고 생각하면 된다. 
2. cuDNN :CUDA는 그래픽 작업을 위해서 존재하는 것이고, cuDNN은 딥러닝 라이브러리를 위해서 추가된 기술.

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/91636912-636abc00-ea3f-11ea-8fce-7ae044abe32b.png' alt='drawing' width='300'/></p>

3. 위의 모든 것이 잘 설치되어 있다면, 당연히    
    $ nvidia-smi    
    $ watch -n 1 nvidia-smi  
4. 학습을 하는 동안 GPU 메모리 초과가 나오지 않게 조심해야 한다.   
    따라서 GPU를 사용하는 다른 Processer를 끄는 것을 추천한다. (nvidia-smi에 나옴)

# 4. Object Detection 개요    

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/91637286-6e731b80-ea42-11ea-9841-5463bd8cf10c.png' alt='drawing' width='600'/></p>

- Feature Extraction Network : Backbone Network, Classification을 위해 학습된 모델을 사용한다. 핵심적인 Feature를 뽑아내는 역할을 한다. 
- Object Detection Network 
- Region Progosal : Selective Search, RPN 등등.. 

- Image Resolution, FPS, Detection 성능의 당연한 상관 관계 아래 그림 참조    
  yoloV2 성능비교를 통해서 확인해 보자. 
<p align="center"><img src='https://user-images.githubusercontent.com/46951365/91637407-27395a80-ea43-11ea-9952-26a648096738.png' alt='drawing' width='600'/></p>


  