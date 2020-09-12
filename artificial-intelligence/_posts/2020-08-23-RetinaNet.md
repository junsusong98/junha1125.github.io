---
layout: post
title: 【Paper】 RetinaNet - Focal Loss, FPN
description: >
     RetinaNet에 대해서 차근 차근 핵심만 파악해보자.
---

FPN에 대한 자세한 내용 정리는 제가 이전에 정리해놓은 [FPN 게시물](https://junha1125.github.io/artificial-intelligence/2020-03-18-paper-Tstory3/)을 참고하시면 매우 좋습니다. 

# 1. RetinaNet 특징
- 2017 Facebook AI Researcher Team 
- 1 stage detector는 2 stage detector 보다 정확도가 떨어진다는 분위기가 있었다. 
- 이러한 인식을 해소했다. 더 빠르고 더 정확한 Detection이 가능하다.
- 지금까지도 수행성능이 다른 Detection 모델 보다 뛰어나다. 
- Focal Loss(새로운 Loss 함수. Cross Entropy 대체), FPN(BackBone으로 사용)을 적극 사용했다. 

- 성능 : COCO에 대한 수행성능이 좋다 = 작은 객체에 대해서도 잘 찾는다. box를 정확하게 그린다.

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/92297653-9b608a80-ef7c-11ea-96d9-160e8871d5be.png' alt='drawing' width='600'/></p>


# 2. Focal Loss
**"잘 Detect할 수 있는 '배경이나, 큰 Object'에 대한 모델의 학습을 살살시키고, 
    어럽게 Detect 해야 하는 '작은 Object나 변형 형태의 Object'에 대한 모델의 학습은 크게  일어나게 함으로써, 찾기 쉬운 것도 잘 찾고 찾기 어려운 것도 잘 찾도로 만들자!"**

1. Focal Loss는 Cross Entropy의 변형이다. Cross Entropy 공식 : 

    <p align="center"><img src='https://user-images.githubusercontent.com/46951365/92298611-e0d58580-ef85-11ea-8d20-e81b3af0113c.png' alt='drawing' width='600'/></p>

2. Cross Entropy 의 문제점 
    - "모델이 이미 잘 Detect할 수 있는 것에 더 잘 Detect할 수 있도록 만든다. 모델이 어렵게 Detect하는건 계속 어렵게 Detect하게 하도록 만든다."

    <p align="center"><img src='https://user-images.githubusercontent.com/46951365/92298157-175cd180-ef81-11ea-8766-2571cb426eee.png' alt='drawing' width='800'/></p>

3. Why?? 왜 그런 문제점이 생길까? - **Class Imbalance 문제**
    - 위의 사진에서 왼쪽 강아지가 Easy Example, 오른쪽 강아지가 Hard Example 이다. 
    - 아래 중간 내용 보충  
        1. 많은 Background(negative ROI) Easy Example가 이미지의 대부분이다. 
        2. 적은 foreground (Object=positive ROI) Hard Example는 매우 유용한 정보이고, 신경망 모델은 이것을 위해 학습을 해나가야 한다. 
        3. 하지만 아래의 설명처럼 Easy Example에 대한 Loss가 너무 압도적으로 크다. 그래서 위의 사진의 Cross Entropy로 학습할 때의 현상이 나타나게 된다. 
    <p align="center"><img src='https://user-images.githubusercontent.com/46951365/92298286-28f2a900-ef82-11ea-81f4-3dbf15c9336c.png' alt='drawing' width='800'/></p>

4. Class Imbalance 해결 방안
    - 기존의 One Stage Detector (Faster-RCNN 에서 배운 방법) :
        - 학습 시키기 위한 DataSet을 그냥 다 집어넣는게 아니라, Foreground와 Backround의 비율을 같게하여 신경망에 집어넣어 학습시킨다. 
        - 작은 Object는 Crop하고 확대하는 등의 Data Augmentaion에 집중했다. 

    - Cross Entropy를 바꾸자! Focal Loss로! 
        - CE = Cross Entropy,  FL = Focal Loss
        - 이미 높은 confidence로 예측하는 것의 Loss는 많이~~!! 낮추고, 
        - 낮은 confidence로 예측하는 것의 Loss는 조오금... 낮춘다

    <p align="center"><img src='https://user-images.githubusercontent.com/46951365/92298586-6a388800-ef85-11ea-9c63-20101c371d08.png' alt='drawing' width='800'/></p>



# 3. Featur Pyramid Network
- FPN에 대한 자세한 내용 정리는 제가 이전에 정리해놓은 게시물을 참고하시면 좋습니다.   
    - [FPN PPT 발표 자료](https://junha1125.github.io/artificial-intelligence/2020-03-18-paper-Tstory3/)
    - [논문을 직접 간략히 구현해 놓은 코드](https://junha1125.github.io/artificial-intelligence/2020-03-18-paper-Tstory3_code/) 
- 위의 게시물에서 추가적인 내용만 기록해 놓겠습니다. 

1. FPN
    1. (a) : Sliding window 방식에서 사용하는 방법
    2. (b) : CNN 맨 위에 Sementic한 Feature를 가지지만 너무 Resolution작음
    3. (c) : SSD 에서 사용하는 방법
    4. (d) : FPN

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/92300532-96103980-ef96-11ea-99db-8087adbed3d3.png' alt='drawing' width='500'/></p>

2. Upsampling + Lateral connetction 이후 3 x 3 conv를 하는 이유 
    - upsampling의 불완전한 feature imformation 복원을 해소시켜주기 위해서.
    - aliasing effect 문제를 해결하기 위해 - Skip connection을 그냥 하면, 서로 다른 Signal이 섞여 혼동되어 본래 자신들의 의미가 서로 사라지는 문제가 있다.(?) 3 x 3 conv를 통해서 그런 문제를 해결 했다고 한다.

3. FPN과 RetinaNet 
    - 9개의 anchor box가 FPN에서 나온 P2~P5의 개별 Layer의 개별 Grid에 할당된다, 
    - anchor box는 3개의 서로다른 크기와 3개의 서로 다른 스케일을 가진다. 
    - 약 100k개의 anchor box들이 생긴다. 
    - 개별 anchor(A)는 아래의 이미지 처럼 K와 4에 대한 정보를 가진다. (K개의 클래스 확률값과 Bounding box regression 4개의 좌표)

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/92300896-af66b500-ef99-11ea-9616-d802a3045856.png' alt='drawing' width='600'/></p>