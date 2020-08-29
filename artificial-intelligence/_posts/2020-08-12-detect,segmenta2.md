---
layout: post
title: 【Vision】 Detection과 Segmentation 다시 정리 2
description: >  
    당연하다고 생각하지만, 아직은 공부할게 많은 Detection과 Segmentation에 대한 개념을 다시 상기하고 정리해보면서 공부해볼 계획이다. 

---
Detection과 Segmentation 다시 정리 2 
  
# Dataset에 대한 진지한 고찰

## 1. Pascal VOC
- XML Format (20개 Class)
- 하나의 이미지에 대해서 Annotation이 하나의 XML 파일에 저장되어 있다. 
- [데이터셋 홈페이지 바로가기](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#data)
- 대회 종류 
    - Classification/Detection
    - Segmentation
    - Action Classification
    - Person Layout
- Annotation : 

    <img src = 'https://user-images.githubusercontent.com/46951365/91626099-1d820980-e9e7-11ea-9932-5874e2a2ff79.png' alt='drawing' width='600'/>

## 2. MS COCO
- json Format (80개 Class)
- 모든 이미지에 대해서 하나의 json 파일이 존재한다. 
- Pretrained Weight로 활용하기에 좋다. 


## 3. Google Open Image
- csv Format (600개 Class)
- Size도 매우 크기 때문에 학습에도 오랜 시간이 걸릴 수 있다. 