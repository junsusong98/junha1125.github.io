---
layout: post
title: 【Vision】 Detection과 Segmentation 다시 정리 1
description: >  
    당연하다고 생각하지만, 아직은 공부할게 많은 Detection과 Segmentation에 대한 개념을 다시 상기하고 정리해보면서 공부해볼 계획이다. 

---
Detection과 Segmentation 다시 정리1

## 1. Object Detection 계보
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


## 2. Object Detection 
- 요소
    - Region Proposal
    - Feature Extraction & Network Prediction (Deep Nueral Network)
    - IOU/ NMS/ mAP/ Anchor box
- 난제 
    - 하나의 이미지에서 여러개의 물체의 Localization + classification해야함
    - 물체의 크기가 Multi-Scale Objects 이다. 
    - Real Time + Accuracy 모두를 보장해야함. 
    - 물체가 가려져 있거나, 물체 부분만 나와 있거나
    - 훈련 DataSet의 부족 (Ms Coco, Google Open Image 등이 있다.)  

## 3. Object Localization 개요  
- 아래와 같은 전체적인 흐름에서 Localizattion을생각해보자.   
    - <img src='https://user-images.githubusercontent.com/46951365/91536084-27582e00-e94f-11ea-974f-03e4be5913fa.png' alt='drawing' width="600"/>
    - 위 사진의 4개의 값을 찾는다. (x1, y1, width, hight) 물체가 있을 법한 이 좌표를 찾는 것이 목적이다. 
    - Object Localization 예측 결과  
        - class Number, Confidence Score, x1, y1, width, hight  
    - 2개 이상의 Object 검출하기  
        - Sliding Window 방식 - anchor(window)를 슬라이딩 하면서 그 부분에 객체가 있는지 계속 확인하는 방법. 다양한 크기 다양한. 비율의 windows.   
        또는 이미지를 조금씩 작게 만든 후 상대적으로 큰 window를 슬라이딩 하는 방식도 있다. (FPN의 기원이라고 할 수 있다.)  
        - Region Proposal : 위와 같은 방법이 아니라, 일종의 알고리즘 방식으로 물체가 있을 법한 위치를 찾자.  
            1. [Selective Search](https://donghwa-kim.github.io/SelectiveSearch.html) : window방법보다는 빠르고 비교적 정확한 추천을 해줬다. Pixel by Pixel로 {컬러, 무늬, 형태} 에 따라서 유사한 영역을 찾아준다. 처음에는 하나의 이미지에 대략 200개의 Region을 제안한다. 그 각 영역들에 대해 유사한 것끼리 묶는 Grouping 과정을 반복하여 적절한 영역을 선택해 나간다. (Pixel Intensity 기반한 Graph-based segment 기법에 따라 Over Segmentation을 수행한다.)
            <img src='https://user-images.githubusercontent.com/46951365/91543970-fb8e7580-e959-11ea-868c-ebd4d8b58daf.png' alt='drawing' width="600"/>
            2. 