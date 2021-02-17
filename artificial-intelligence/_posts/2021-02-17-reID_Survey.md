---
layout: post
title: 【Re-ID】Person Re-identification A Survey and Outlook
---

- **논문** : [Deep Learning for Person Re-identification: A Survey and Outlook](https://arxiv.org/pdf/2001.04193.pdf)
- **분류** :  Re-identification
- **저자** : Mang Ye, Jianbing Shen
- **읽는 배경** : 연구실 과제 참여를 위한 선행 학습



# 1. Abstract

1. Person re-ID란? 겹치지 않는 다중 카메라들 사이에 person of interest를 검색하는 것(retrieving) 
2. 3개의 다른 perscpecitve를 가지는 **closed world Person-Re-ID (상용화를 위한 것이 아닌 연구 단계의 연구들 = research-oriented scenarios**)에 대한 종합적 개요가 있다. with (1) in-depth analysis (2) deep feature representation learning (3) deep metric learning  (4) ranking optimization. 이 연구는 이미 거의 포화 상태다. 
3. 그래서 5개의 다른 perscpective를 가지는 **Open-world setting으로 넘어가고 있다. (상용화를 위한 연구 단계의 연구 = practical applications)** 
4. 이번 논문에서 새로 소개하는 것은 이런 것이 있다.
   1. a powerful AGW baseline
   2. 12개의 Dataset
   3. 4개의 다른 Re-ID task
   4. e a new evaluation metric (mINP) 



# 2. Introduction

\<서론\>

1. 초기에는 a specific person를 찾기 위함이었다. 시대의 발전과 공공 안전 중요도의 증가, 지능형 감시 시스템 need에 의한 기술 발전이 이뤄지고 있다.
2. Person Re-ID의 Challenging task (방해요소들) : [the presence of different viewpoints [11], [12], varying low-image resolutions [13], [14], illumination changes [15], unconstrained poses [16], [17], [18], occlusions [19], [20], heterogeneous modalities [10], [21], complex camera environments, background clutter [22], unreliable bounding box generations, etc. the dynamic updated camera network [23], [24], large scale gallery with efficient retrieval [25], group uncertainty [26], significant domain shift [27], unseen testing scenarios [28], incremental model updating [29] and changing cloths [30] also greatly increase the difficulties.] 이런 요소들이 있기 때문에 연구들은 더욱 더 이뤄져야 한다.
3. 딥러닝 이전에는 이 기술을 사용했다. the handcrafted feature construction with body structures [31], [32], [33], [34], [35] or distance metric learning [36], [37], [38] 
4. **딥러니 이후에는 [5], [42], [43], [44] 이 논문이 놀라운 성능을 내었다**
5. 이 논문의 특별한 차별점
   1. powerful baseline (**AGW: Attention Generalized mean pooling with Weighted triplet loss**) 제시한다.
   2. new evaluation metric (mINP: mean Inverse Negative Penalty) 제시한다. mINP는 현재 존재하는 지표인 CMC/mAP를 보충하는 것으로써, **정확한 matches를 발견하는 cost를 측정**한다.

---

\<본론\>

![image-20210217132329680](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210217132329680.png)

**일반적인 Re-ID 시스템은 아래의 5 절차를 거친다.**

1. Raw Data Collection : 감시 카메라로 부터 영상 받아오기
2. Bounding Box Generation : Detecting, Tracking Algorithms 이용하기
3. Training Data Annotation : Close world에서는 Classification 수행하기. Open world에서는 unlabel classification 수행하기
4. Model Training : Re-ID 수행하기 (다른 카메라, 같은 Label data를 묶고, 전체를 Gallery에 저장하기. 
5. Pedestrian Retrieval : Query person 검색하기(찾기). 여기서  query-to-gallery similarity를 비교해서 내림차순으로 나열하는 A retrieved ranking를 수행한다. 이 작업을 위해 retrieval performance를 향상시키기 위한, the ranking optimization을 수행해야 한다.

---

![image-20210217132347553](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210217132347553.png)

**Closed-world 에서 Open-world Person Re-ID 으로의 필요한 핵심 포인트**

1. ingle-modality vs. Heterogeneous Data : 지금은 카메라 영상을 이용하는데, infrared images [21], [60], sketches [61], depth images [62] 등을 활용한 Re-ID연구와 사용이 필수적이다.
2. Bounding Box Generation vs. Raw Images/Videos : close world에서는 "Search + Predict + Bouning Box는 이미 되었다"고 가정하고 Re-ID를 수행한다. 하지만 이제는 같이 end-to-end person search가 필요하다.
3. Sufficient Annotated Data vs. Unavailable/Limited Labels : label classification은 실생활에서 불가능하다. limited labels이기 때문이다. 따라서 unsupervised and semi-supervised Re-ID 연구가 필요하다. 
4. Correct Annotation vs. Noisy Annotation : close world에서는 정확한 Bounding Box를 가정한다. 하지만 실제 Detection 결과는 부정확하고 Noise가 있다. noise-robust person Re-ID를 만드는 연구가 필요하다.
5. Query Exists in Gallery vs. Open-set : close world에서는 Gallery에 query person이 무조건 존재한다고 가정한다. 하지만 없을 수도 있는거다. 검색(retrieval)보다는 the verification(존재 유무 확인 및 검색)이 필요하다.

---

\<결론\>







