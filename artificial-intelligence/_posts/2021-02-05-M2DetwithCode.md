layout: post
title: 【Detection】Understanding M2Det paper with code w/ my advice

- **논문** : [M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network](https://arxiv.org/abs/1811.04533)

- **분류** : Object Detection

- **저자** : Qijie Zhao , Tao Sheng , Yongtao Wang∗ , Zhi Tang

- **읽는 배경** : Recognition Basic. Understand confusing and ambiguous things.

- **읽으면서 생각할 포인트** : 코드와 함께 최대한 완벽히 이해하기. 이해한 것 정확히 기록해두기.

- **느낀점**  : 

  - 최근 논문을 많이 읽어서 느끼는 점은, **각 논문은 특별한 목표가 1,2개 있다.** (예를 들어 Cascade는 close false positive 객체를 줄이기 위함이 목표이고, FPN는 Multi-scale object detect를 목적으로 한다.) 그리고 그 목표 달성을 위해, 이전 신경망들의 정확한 문제점을 파악한다. 그 문제점을 해결하기 위해 새로운 아이디어를 첨가한다. Cascade에서 배운 것 처럼, 복잡한 아이디어를 마구 넣는다고 성능이 좋아지지 않는다. **<u>핵심은 목표를 확실히 잡고, 문제점을 확실히 분석하고, 그것을 해결하기 위해 아이디어를 넣어보는 일렬이 과정이 정말 중요한 것 같다.</u>**
  
  



# 1. M2Det Paper Review 

## 1. Conclusoin

- Multi-Level Feature Pyramid Network (**MLFPN**) for <u>different scales</u>
- <u>New modules</u>
  1. Multi-lebel features by **Feature Fusion Module (FFM v1)**
  2. **Thinned U-shape Modules (TUMs)** + **Fature Fusion Modules (FFM v2s)**
  3. multi-level multi-scale features (= 즉! **the decoder layers of each TUM**)
  4. multi-level multi-scale features  With the same scale (size) by a **Scale-wise Feature Aggregation Module (SFAM)**
  5. SOTA Among the one-stage detectors on MS-COCO



## 2. Abstract, Introduction

- Scale-variation problem -> FPN for originally <u>classification</u> tast
- MLFPN construct more effective feature pyramids then FPN for <u>Objec Detection tast</u>



## 3.Introduction

- **conventional method** for Scale variation
  1. image pyramid at only testing time -> memory, computational complexity 증가
  2. feature pyramid at both training and testing phases
- **SSD, FPN** 등등은 for originally <u>classification</u>(backbone이니까) tast -> 한계가 존재한다. 
  1. 그냥 backbone 중간에서 나온 형태 이용 - 정보 불충분 - object detection 하기 위해 정보 불충분
  2. FPN의 p1, p2, p3는 Single-Level Feature 이다. 우리 처럼 Multi-Level Feature Pyramid가 아니다.
  3. Low level feature ->  simple appearances // high-level feature -> complex appearances. 유사한 scale의 객체는 complex appearances로써 구별되어야 하는데, Single-Level Feature로는 충분하지 않다. 
  4. Related Work : Faster R-CNN, MS-CNN, SSD, FPN, DSSD, YOLOv3, RetinaNet, RefineDet
  5. <img src="C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210205154750949.png" alt="image-20210205154750949" style="zoom: 67%;" />
- **우리의 진짜 목적은**  <u>effective feature pyramid</u> for <u>detecting objects</u> of <u>different scales</u> 이다.
  - 우리 모델은 위의 한계를 해결한다. 어차피 뒤에 똑같은 내용 또 나올 듯.



## 4. Proposed Method

- 꼭 아래의 그림을 새탭으로 열어서 이미지를 확대해서 볼 것. 논문에 있는 내용 중 중요한 내용은 모두 담았다. 정성스럽게 그림에 그리고 필기해놨으니, 정성스럽게 참고하고 공부하기.

![image-20210205165329776](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210205165329776.png)

![image-20210205165329777](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210205165329777.png)

- <u>MLFPN 내용 추가</u>
  - FFMv1 : VGG에서 나오는 2개의 semantic Feature map 융합. FPN은 여기서 low feature map data를 사용하는데, 이건 충분한 정보가 아니다. 그 feature map은 classification을 위한 feature이지, object detection을 위한 feature이 아니다. 
  - TUM : several feature maps 생성
    - first TUM : learn from X_base only. AND second ... third.. 8th.
    - <img src="C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210205165329775.png" alt="image-20210205165329775" style="zoom:80%;" />
    - L = \# Levels, (=8) = \# TUM. 어찌보면 여기가 진짜 FPN와 유사한 작업이 수행된다. 
    - 마지막에 1x1 conv를 수행해서, smoothness를 추가한다. 
- <u>SFAM</u>
  - Channel-wise Attention Module (Hu, Shen, and Sun 2017) 
    - In order to Encourage features to focus on channels (that they benefit most) - 가장 중요한 channel에 집중! 
    - global average pooling 그리고  excitation step(2개의 nn.Linear Layer)
- <u>Network Configurations</u>
  - To reduce the number of parameters, FFMv2를 통과하고 나오는 Channel을 256으로 설정했다. 이 덕분에 GPU로 학습하기 쉬워졌다. 
  - Original input size는 320, 512, 800 을 받을 수 있게 설정하였다. 
- <u>Detection Stage</u>
  - 6 scale 즉 Multi-Level Feature Pyramid에서, 2개의 Convolution layer를 통과시켜서,  location regression and classification 정보를 얻어 내었다. 
  - default(anchor) boxes 설정 방법은 SSD를 따랐다. (2개 size + 3개 ratios = 6 Anchor). 
  - Anchor 중에서 (Like objectness score) 0.05이하의 anchor들은 low score로써 Filter out 시켜버렸다. (아에 Backgound라고 판단. 학습(nagative mining) 및 추론에 사용 안 함)
  - 0.01까지 낮춰 보았지만, 너무 많은 detection results발생으로, inference time에 영향을 크게 주어서 0.05를 사용하였다.  
  - 그리고 Input size가 800x800이면, scale ranges를 6에서 조금 더 늘렸다. 
  - 더 정확하게 BB를 남기기 위해서, soft-NMS (Bodla et al. 2017) 를 사용했다. 



## 5. Experiments



# 2. qijiezhao/M2Det

1. Github Link : [qijiezhao/M2Det](https://github.com/qijiezhao/M2Det) - 저자가 직접 pytorch사용해서 모델 구현

3. 














