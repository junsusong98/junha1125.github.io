---
layout: post
title: 【Vision】 Detection과 Segmentation 다시 정리 1 - 계보 및 개요, mAP
description: >  
    당연하다고 생각하지만, 아직은 공부할게 많은 Detection과 Segmentation에 대한 개념을 다시 상기하고 정리해보면서 공부해볼 계획이다. 

---
Detection과 Segmentation 다시 정리 1

## 1. Object Detection 계보
1. Pascal VOC 데이터 기반에서 AlexNet을 통해서 딥러닝이 화두가 되었다. 
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
            2. RPN(Region Proposal Network)
    

## 4. Object Detection 필수 구성 성분  
- IOU : 2개의 Boundiong Box에 대해서 (교집합 / 합집합) 

- NMS(Non Max Suppression) : Object가 있을 만한 곳을 모든 Region을 배출한다. 따라서 비슷한 영역의 Region proposal이 나올 수 있다. 일정 IOU값 이상의 Bounding Boxs에 대해서 Confidence Score가 최대가 아닌것은 모두 눌러버린다. 
    1. Confidence Score 가 0.5 이햐인 Box는 모두 제거
    2. Confidence Score에 대해서 Box를 오름차순 한다
    3. 높은 Confidence Score의 Box부터 겹치는 다른 Box를 모두 조사하여 특정 IOU 이상인 Box를 모두 제거 한다(IOU Threshold > 0.4, **[주의]** 이 값이 낮을 수록 많은 Box가 제거 된다. )
    4. 남아있는 Box만 선택



## 5. 필수 성능지표 mAP (mean Average Precision)
1. Inference Time도 중요하지만 AP 수치도 중요하다.     

2. Precision(예측 기준)과 Recall(정답 기준)의 관계는 다음과 같다.   
    <img src = 'https://user-images.githubusercontent.com/46951365/91571417-4ff91b80-e981-11ea-8bbb-aee275642e62.png' alt ='drawing' width='700'/> 

3. 또 다른 설명은 여기([참고 이전 Post](https://junha1125.github.io/blog/projects/2020-01-13-ship_5/))를 참고 할 것. 

4. Precision과 Recall이 적절히 둘다 좋아야 좋은 모델이다. 라고 할 수 있다. 

5. Pascal VOC - IOU:0.5  // COCO challenge - IOU:0.5 0.6 0.7 0.8 0.9   

6. TN FN FP TP 분류하는 방법  
    <img src = 'https://user-images.githubusercontent.com/46951365/91576928-fa257300-e982-11ea-9b99-3bde26dd4539.png' alt ='drawing' width='700'/>   
    <img src = 'https://user-images.githubusercontent.com/46951365/91577133-42449580-e983-11ea-9056-8d92f463f5a7.png' alt ='drawing' width='700'/>    

7. 암인데 암이 아니라고 하고, 사기인데 사기가 아니라고 하면 심각한 문제가 발생하므로 '<u>진짜를 진짜라고 판단하는 **Recall**</u>'이 중요하다.(FN이 심각한 문제를 야기한다.) 반대로 스팸 메일이 아닌데 스팸이라고 판단하면 심각한 문제가 발생하므로 '<u>내가 진짜라고 판단한게 진짜인 **Precision**</u>'이 중요하다.(FP가 심각한 문제를 야기한다.)    

    <img src = 'https://user-images.githubusercontent.com/46951365/91577753-22fa3800-e984-11ea-9c8a-7369f72fbeea.png' alt ='drawing' width='700'/>    

8. 이러한 조절을 Confidence Threshold를 이용해서 할 수 있다.   
Ex. Confidence Threshold를 낮게 한다면 Recall(재현율)이 높아 진다. (다 Positive라고 판단해버리기)    
Confidence Threshold을 높게 한다면 Precision(정밀도)가 높아진다.(정말 확실한 경우만 Positive라고 예측하기)    
    <img src = 'https://user-images.githubusercontent.com/46951365/91578181-cba89780-e984-11ea-8e1d-7d0cd0ead048.png' alt ='drawing' width='600'/>     
    <img src = 'https://user-images.githubusercontent.com/46951365/91580069-83d73f80-e987-11ea-8039-5b947f3b8b90.png' alt ='drawing' width='600'/>     

9. 즉 Confidence에 따른 Precison과 Recall의 변화 그래프이므로, 여기([참고 이전 Post](https://junha1125.github.io/blog/projects/2020-01-13-ship_5/))에서 Confidence에 대해서 내림차순 정렬을 하고, 차근차근 Recall, Precision점을 찍고 그 그래프의 넓이를 구하는 것이다. 

10. Confidence Threshold가 큰것부터 시작했으므로, Precision은 높고, Recall은 낮은 상태부터 시작한다. 주의할 점은 오른쪽 최대 Precision 값을 연결하고, mAP를 계산한다!   
    <img src = 'https://user-images.githubusercontent.com/46951365/91580875-ad449b00-e988-11ea-9b65-f1efdd835aa4.png' alt ='drawing' width='400'/>

11. 지금까지가 AP를 구하는 방법이었다. 즉 AP는 한개의 Object에 대해서 값을 구하는 것이다. ([참고 이전 Post](https://junha1125.github.io/blog/projects/2020-01-13-ship_5/)) 그리고 모든 Object Class(새, 나비, 차, 사람 등등)에 대해서 AP를 구한 후 평균값을 사용하는 것이 바로 mAP이다.  

12. COCO Dataset에 대해서는 IOU Threshold를 다양하게(AP@\[.50:.05:.95]) 주기 때문에 높은 IOU에 대해서 낮은 mAP가 나올 수 있음을 명심해야 한다.(높은 IOU Threshold라는 것은 FP와 TP 중 TP가 되기 힘듦을 의미한다.)   
그리고 COCO는 Object의 Scale 별로 대/중/소에 따른 mAP도 즉정한다.    
    <img src = 'https://user-images.githubusercontent.com/46951365/91581239-29d77980-e989-11ea-89e4-aad3b2f95f7b.png' alt ='drawing' width='600'/>



​    