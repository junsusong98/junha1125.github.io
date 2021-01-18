---
layout: post
title: 【Paper】 Deep Learning for Generic Object Detection, A Survey - Summary
---
Deep Learning for Generic Object Detection, A Survey 논문 리뷰 및 정리

## Deep Learning for Generic Object Detection: A Survey

## My Conclusion after I read the paper.

- this paper introduce Either too easy or too old Detector for me... Rather read the latest paper.
  - [M2det](https://arxiv.org/pdf/1811.04533.pdf)
  - [EfficientDet](https://arxiv.org/pdf/1911.09070.pdf)
  - [YOLOv4](https://arxiv.org/pdf/2004.10934.pdf)
  - [RefineNet](https://arxiv.org/pdf/1611.06612.pdf)


## 0. Abstract

1.  a comprehensive survey of the recent achievements 
2.  More than 300 research contributions
3. frameworks/ representation/ proposal generation/ context modeling/ training strategies/ evaluation metrics
4. promising directions for future research



## 1. introduction

- Deep Convolutional Neural Network (DCNN) called AlexNet in 2012 
- Over the past 5 years, this article attempts to track in order to gain a clearer picture 
-  popular datasets/ evaluation metrics/ context modeling/ detection proposal methods.

- many object detection previous surveys (refer to reference this paper later)
  - pedestrian detection
  - face detection
  - vehicle detection
  - traffic sign detection
  - generic object detection
- we have limited our focus to top journal and conference papers. and limited picture detection not Video.
- follows
  1. 20-years are summarized
  2. A brief introduction to deep learning
  3. Popular datasets and evaluation criteria
  4. the milestone object detection frameworks
  5. state-of-the- art performance
  6. future research directions



## 2. Generic Object Detection

2.1 The Problem

- future challenges will move to the pixel level object detection.

2.2 Challenges

- Accuracy Related Challenges 
  - intra-class variations : color, texture, material, shape, size, poses, clothes
  - Imaging condition variations : lighting,  physical location, weather, backgrounds
  - 20, 200, 91 object classes / VOC, ILSVRC, COCO is much smaller than can be recognized by humans.
- Efficiency and Scalability Related Challenges
  -  mobile/wearable devices have limited computational capabilities and storage space
  - A further challenge is that of scalability : unseen objects, unknown situations, it may become impossible to annotate them manually, forcing a reliance on weakly supervised strategies.

2.3 Progress in the Past 2 Decades

- <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210114173125275.png?raw=tru" alt="image-20210114173125275" style="zoom: 80%;" />
- accurate annotations are labor intensive to obtain
- Ability to detect many object categories matches that of humans(인간 같은) 3000~3000 categories  is undoubtedly an unresolved problem so far.



## 3.  A Brief Introduction to Deep Learning

- pass



## 4.1 Dataset

- PASCAL VOC
- ImageNet
- MS COCO
- Places
- Open Images



### 4.2 Evaluation Criteria

- Frames Per Second (FPS)
- precision, and recall.
- Average Precision (AP) -> over all object categories, [the mean AP (mAP)](https://junha1125.github.io/artificial-intelligence/2020-08-10-detect,segmenta/)



## 5. Basic Frameworks

- <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210114192428720.png?raw=tru" alt="image-20210114192428720" style="zoom:67%;" />
- <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210114201752611.png?raw=tru" alt="image-20210114201752611" style="zoom: 80%;" />

