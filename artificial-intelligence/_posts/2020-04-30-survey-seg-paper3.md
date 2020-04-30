---
3layout: post
title: (논문) Image Segmentation Using Deep Learning -A Survey [3]
description: >  
    Image Segmentation Using Deep Learning: A Survey 논문 리뷰 및 정리
---

(위성Segment) Segmentation Survey 논문 정리 3
논문 원본 : **2020 (cite:6) [Image Segmentation Using Deep Learning: A Survey](https://arxiv.org/abs/2001.05566)** 



# Section 4: IMAGE SEGMENTATION DATASETS

- 이 세션에서는 이미지 세그먼테이션 데이터 세트에 대한 요약을 제공한다
- 2가지 범주로 분류 : 2D 이미지, 3D 이미지, 2.5 이미지(depth카메라)
- 일부 데이터셋에서는 data augmentation을 하는 것이 좋다.(특히 의료 이미지에서) 그 방법으로는  reflection, rotation, warping, scaling, color space shifting, cropping, and projections onto principal components과 같은 방법이 있다. 
- data augmentation를 통해서 모델의 성능을 향상시키고, 수렴 속도를 높히고, Overfitting 가능성을 줄이며, 일반화 성능을 향상시키는데 도움이 될 수 있다. 



## 4.1 - 2D Datasets

1. PASCAL Visual Object Classes (VOC) [141]
   - 5가지 정보 : classification, segmentation, detection, action recognition, and person layout
   - 21가지 segmentation class 
   - 각각 1,464, 1,449개의 Train, Validation 이미지
2. PASCAL Context [142]
   - PASCAL VOC의 확장데이터 
   - 400개의 segmentation class {divided into three categories (objects, stuff, and hybrids)} 
   - 하지만 너무 희박하게 존제하는 data 때문에 사실상, 59개의 자주 나타나는 class로 일반적으로 적용된다.
3. Microsoft Common Objects in Context (MS COCO) [143] 
   - dataset  : 91 objects types / 2.5 million labeled instances / 328k images
   - segmenting individual object instances
   - detection : 80 classes / 82k train images /  40.5k validation / 80k test images
4. Cityscapes [144]
   - semantic understanding of urban street scenes에 집중되어 있다.
   - diverse set of stereo video sequences / 50 cities / 25k frames
   - 30 classes(grouped into 8 categories) 
5. ADE20K[134] / MIT Scene Parsing (SceneParse150)
   - scene parsing algorithms을 위한 training and evaluation platform
   - 20K images for training/ 2K images for validation /  150 semantic categories
6. SiftFlow [145]
   -  2,688개의 정보 있는 이미지
   - 258*258이미지
   - 8 가지 풍경(산 바다 등.. ) / 33 semantic classes
7. Stanford background [146]
   - 하나 이상의 forground(객체)가 있는 715 images
8. BSD (Berkeley Segmentation Dataset) [147]
   - 1,000 Corel dataset images 
   - empirical basis(경험적 기초) for research on image segmentation
   - 공개적으로 흑백이미지, 칼라이미지 각각 300개가 있다.
9. Youtube-Objects [148], [149]
   - 10 개의 PASCAL VOC class에 대한 동영상 데이터(weak annotations) [148] 
   - 10,167 annotated 480x360 pixel frame [149]
10. KITTI [150]
    - mobile robotics and autonomous driving에서 자주 쓰이는 데이터 셋
    - hours of videos(high-resolution RGB, grayscale stereo cameras, and a 3D laser scanners)
    -  original dataset에서는 ground truth for semantic segmentation를 제공하지 않는다.
    - 그래서 직접 annotation한 데이터 셋으로 [151]에서 323 images from the road detection challenge with 3 classes(road, vertical(빌딩, 보행자 같은), and sky)
11. 그 외 추가 데이터 셋
    - Semantic Boundaries Dataset (SBD) [152]
    - PASCAL Part [153]
    - SYNTHIA [154]
    - Adobes Portrait Segmentation [155]

![image](https://user-images.githubusercontent.com/46951365/80694578-a9565880-8b0f-11ea-85e3-3744f60acc22.png)



<br>

## 4.2 - 2.5D Datasets































