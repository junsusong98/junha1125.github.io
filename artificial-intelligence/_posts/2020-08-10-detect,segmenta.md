---
layout: post
title: 【Vision】 Detection과 Segmentation 다시 정리 1
description: >  
    당연하다고 생각하지만, 아직은 공부할게 많은 Detection과 Segmentation에 대한 개념을 다시 상기하고 정리해보면서 공부해볼 계획이다. 

---
Detection과 Segmentation 다시 정리1

## 1. Object Detection
1. Pascal VOC 데이터 기반에서 알렉스넷을 통해서 딥러닝이 화두가 되었다. 
2. Detection을 정확히 분류하면 다음과 같이 분류할 수 있다. 
    - Classification
    - Localization 
    - Detection : Bounding Box Regression + Classification
    - Segmentation : Bounding Box Regression + Pixel Level Classification
3. 계보   
    - <img src='https://user-images.githubusercontent.com/46951365/91528239-443a3480-e942-11ea-9051-fb35d86b2f5c.png' alt='drawing' width="600"/>
    - Traditional Detection 알고리즘 - VJ det, HOG det ...
    - Deep Learning Detection 알고리즘 - 1 Stage Detection, 2 Stage Detectio (Region proposal + Classification)  
    - SSD -> Yolo2, Yolo3
    - Retina-Net : 실시간성은 Yolo3보다는 아주 조금 느리지만, 정확성은 좋다.
4. Detection은 API가 잘 정해져있지만, Train에 오류가 많이 나서 쉽지가 않다.   


## 2. Object Detection 구성 요소