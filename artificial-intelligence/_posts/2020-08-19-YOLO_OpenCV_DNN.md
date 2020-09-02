---
layout: post
title: 【Python-Module】 OpenCV DNN 모듈을 사용해서 YOLO 수행하기 
description: >
     주의 사항 : 지금까지의 OpenCV DNN 모듈 이용과는 흐름이 다르다.      
     이전 Post를 통해서 YOLO의 이론에 대해서 공부했다. OpenCV DNN 모델을 사용해서 YOLO를 수행해보자. 
---

OpenCV DNN 모듈을 사용해서 YOLO 수행하기   
주의 사항 : 지금까지의 OpenCV DNN 모듈 이용과는 흐름이 다르다.    

###  참고 파일 List 
(conda tf113 환경)   
1. /DLCV/Detection/yolo/OpenCV_Yolo_이미지와_영상_Detection.ipynb
2. /DLCV/Detection/yolo/KerasYolo_이미지와_영상_Detection.ipynb
     
# 1. OpenCV DNN Yolo 사용방법

- 핵심 사항
     - 지금까지는 Tensorflow 기반 pretrain 모델을 가져왔다. 
     - Tensorflow에서 Yolo를 지원하지 않는다. Yolo의 창시자 사이트에 가서 weight와 conf파일을 받아와야 한다.

-  DarkNet 구성 환경 및 Yolo 아키텍처에 따라서 사용자가 직접 Object Detection 정보를 추출해야한다. 
     - 아래와 같이 사용자가 Scale이 다른 Feature Map에서 Detection결과 추출과 NMS를 직접 해야한다. OpenCV에서 해주지 않는다. 
     - 82번 94번 106번 (13 x 13, 26 x 26, 52 x 52) Layer에 직접 접근해서 Feature Map정보를 직접 가져와야 한다. 
     - NMS 도 직접 해야한다. OpenCV NMS 함수를 호출해야한다.
     - 이때 Feature Map정보에서 Depth인 4(bounding box regression) + 1(confidence=Objectness Score(객체인지 아닌지)) + 80(class logistic 확률값).
          - 80개의 값 중에서 가장 높은 값을 가지는 위치 인덱스가 객체의 Label값이다. 
          - 4개의 값이 t인 offset값이지 좌표 정보가 아니므로, 좌상단 우하단 좌표로 변환을 해줘야 한다. 

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/91936541-031f9700-ed2b-11ea-8ee3-25fff18896fa.png' alt='drawing' width='500'/></p>

- Pretrained 된 Inferecne 모델 로딩 주의 사항
     - 대부분이 [Darknet 사이트에](https://pjreddie.com/darknet/yolo/)서 직접 weight와 config파일을 다운받와야 한다. 
     - 그것을 통해서 cv.dnn.readNetFromDarknet(confg, weight)  
      (지금까지는 cv.dnn.readNetFromTensorflow(weight, confg) 했었지만.. ) 

- **OpenCV로 Yolo inference 구현 절차 (위의 내용 전체 요약)**
{:.lead}

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/91937226-707ff780-ed2c-11ea-89b2-15356c711a05.png' alt='drawing' width='500'/></p>

# 2. OpenCV DNN Yolo Inference 구현 코드 - 단일 이미지


