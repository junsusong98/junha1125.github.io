---
layout: post
title: 【Paper】 RBox-CNN Rotated Bounding Box 논문 리뷰

# description: > 
    
---

[RBox-CNN Rotated Bounding Box based CNN for ShipDetection in Remote Sensing Image](https://dl.acm.org/doi/10.1145/3274895.3274915)



### **0.Abstract**

\- 이 논문은 RBox-CNN 모델을 제안한다. 

\- Faster RCNN을 배이스로 사용한다.

\- RPN(region proposal network)에서 RRoi(rotation region of interest)를 찾아낸다. 

\- ROI pooling layer에서 diagonal ROI pooling layer도 추가했다.

​     \* 더 좋게 보정된 regression할 rocation을 찾아낸다.

\- 선박 검출 문제에서 좋은 검출 결과를 얻었다. 

\- 항공 사진(원격 탐사 이미지) 객체 검출에 중요한 성과를 얻었다. 

 

### **1.Introduction**

\- 원격 탐사 이미지(remote sensing technology) 사용의 중요성이 증가하고 있다. 이 사진은 항구 관리, 해군 전시 상황에서 많이 사용될 수 있지만, 복잡한 backgorund처리와 밀집된 배들을 처리하기에 어려움을 격고 있었다. 

\- 요즘 핫한 CNN을 사용하고, Faster RCNN을 사용할 것이다.

\- 일반적인 BBox는 배경까지 포함하는 수직 직사각형이므로 정확한 분석이 어렵다. 도한 거기에 NMS을 시행한다면 잡아야할 객체를 잡지 못하는 문제가 발생하기도 한다. 

([NMS](https://jetsonaicar.tistory.com/15) : 객체 중복 검출을 막기위해 겹치는 영역의 BBox들 중에 신뢰도가 가장 높은 Box만 남겨놓는 방법.)

([IoU](https://ballentain.tistory.com/12) : 즉 a영역을 객체라고 분류를 했다면, 그 영역을 ground truth영역과 비교해서 iou가 0.5이상이면, a영역과 ground truth영역에 대한 regression 학습을 시키는 방법. 당연히 분류(클래스)에 대한 학습도 한다. 만약 객체라고 분류한 a영역의 IOU가 0.5이하 이면 그것으로 BBox regression 학습시키지 않는다.) 

\- 따라서 이러한 문제를 해결하기 위해 rotated BBox를 사용 해보고자 했다. 또 diagonal region-of-interest pooling layer 내용도 제안할 예정이다. 

 

 

### **2.Related Work(최근에 시행된 관련 연구들)**

### **3.Methodology(방법론)**

\- Faster R-CNN with rotation anchors 

\- RBox and DRoI pooling llayer

 

### **3.1 Rotated Dounding Box Anchor**

\1. RBox 는 (x,y (center-point), h(짧은 모서리), w(긴 모서리), theta(시계 반대방향으로 돌아간 정도)) 5가지 원소를 가지는 튜플이다. theta는 [-pi/2,pi/2]의 범위를 가진다.



![img](https://k.kakaocdn.net/dn/eAjVoq/btqB6YS6W8m/S8wEu9ICpQgiIrGINvoBa1/img.png)



\2. RBox regression(loss) 는 오른쪽 그림과 같다. 여기서 smooth loss는 Fast-RCNN에서 정의된 내용이다.



![img](https://k.kakaocdn.net/dn/mSFBb/btqB5hsBCiO/CokgUOqqao0esMGdkrbu4k/img.png)

<img src="https://k.kakaocdn.net/dn/ccAMho/btqCTv9pikU/uJ4jgcIh9yPoHhMAHndNQK/img.png" alt="img" style="zoom:50%;" />



Faster rcnn 유투브 강의자료 참조



### 3.2 RoI and DRoI pooling layer

\- 그냥 pooling layer를 사용하는것이 아니고, rotated RoI pooling layer를 사용한다.



![img](https://k.kakaocdn.net/dn/bHydWS/btqB6iYTyZU/bfd3K1JVzOFSCxatHNyYs0/img.png)



\- rotated RoI pooling layer : Box를 수평한 방향으로 다시 회전시킨 후에, Rescale처리를 하는 방법.

(a) : 원래 이미지를 RBox로 예측한 모습

(b) : 회전 후 max pooling적용 -> crop한것 처럼 보임

(c) : diagnal RoI Pooing을 적용. (b)만을 이용한다면, 선박 위의 작은 물체들이 사라지거나 너무 안보이는 현상이 일어난다. 따라서 있는 그대로의 물체를 바라보기 위해 diagnal RoI를 적용한다.

 

### **최종 Achitecture**



![img](https://k.kakaocdn.net/dn/CJOv0/btqB6jwJJyW/cMzVLJE9k2injclxXwKTeK/img.png)



### **3.3 Loss function**



![img](https://k.kakaocdn.net/dn/cvfw6T/btqB79GtBp8/jWLCLcEYfkFKpCpeVXcKAK/img.png)



최종 loss function은 다음과 같이 정의한다. 

이때, p는 예측값 score이다 (softmax처리 한)

​    l은 실제 정답 값 score(one hot)

​    Lloc는 위의 수식에 있다. 

 

### **4 experiments**

### **5 conclusion**



![img](https://k.kakaocdn.net/dn/84V6f/btqCe4Sd5sY/E351bsV1SzjO8neDcUBko1/img.png)

![img](https://k.kakaocdn.net/dn/c5J4TH/btqCcA5UOEK/H8gsb1qXHwU74uf6kUH1ik/img.png)

![img](https://k.kakaocdn.net/dn/rKcBG/btqCdg6WPhW/mjDeB6eVTRxNOMsEwO3jVk/img.png)

![img](https://k.kakaocdn.net/dn/2uI8V/btqCbgG3tIF/ZsN0OFuz9Chb5tIb6wunC1/img.png)

![img](https://k.kakaocdn.net/dn/KZJOD/btqCd1uUsyd/V1AIDW2kik7jKjposra1SK/img.png)

![img](https://k.kakaocdn.net/dn/dFqhmm/btqCcVPwUXA/KvRdxmsGVYMWNh6zwe4sC0/img.png)