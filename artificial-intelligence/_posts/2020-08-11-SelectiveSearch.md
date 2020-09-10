---
layout: post
title: 【Vision】 Selective Search Python Module, IOU 계산 코드 만들기
description: >
  Selective Search Python 모듈을 사용해보고 IOU를 적용해보자.
---

Selective Search Python Module 사용해보기
DLCV/Detection/preliminary/Selective_search와 IOU구하기 파일 참조

1. conda activate tf113
2. jupyter notebook 실행 - \$ nphup jupyter notebook &

## 1. Selective Search 코드 실습

- import cv2 에서 ImportError: libGL.so.1: cannot open shared object file: No such file or directory 에러가 뜬다면  
   \$ sudo apt-get install libgl1-mesa-glx 을 실행하기.

```python
#!pip install selectivesearch
```

```python
import selectivesearch
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

### 오드리헵번 이미지를 cv2로 로드하고 matplotlib으로 시각화
img_bgr = cv2.imread('../../data/image/audrey01.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print('img shape:', img.shape)

plt.figure(figsize=(8, 8))
plt.imshow(img_rgb)
plt.show()

```

    img shape: (450, 375, 3)

- selectivesearch.selective_search()는 이미지의 Region Proposal정보를 반환
- 아래와 같이 selectivesearch 모듈을 사용하는 방법은 다음과 같다.
- 매개변수는 ( 이미지, scale= object의 사이즈가 어느정도 인가? 알고리즘 조정하기, min_size보다는 넓이가 큰 bounding box를 추천해달라)

```python
_, regions = selectivesearch.selective_search(img_rgb, scale=100, min_size=2000)
print(type(regions), len(regions))
print(regions[0])
print(regions[1])
# (x1 y1 좌상단 width hight)  (bounding box size)  (label이 1개면 독자적인 영역. 2개 이상이면 각 Label을 합친 영역이라는 것을 의미)
```

    <class 'list'> 41
    {'rect': (0, 0, 107, 167), 'size': 11166, 'labels': [0.0]}
    {'rect': (15, 0, 129, 110), 'size': 8771, 'labels': [1.0]}

- 반환된 regions 변수는 리스트 타입으로 세부 원소로 딕셔너리를 가지고 있음.
- 개별 딕셔너리내 KEY값별 의미
    - rect 키값은 x,y 좌상단 좌표와 너비, 높이 값을 가지며 이 값이 Detected Object 후보를 나타내는 Bounding box임.
    - size는 Bounding box의 크기.
    - labels는 해당 rect로 지정된 Bounding Box내에 있는 오브젝트들의 고유 ID.
    아래로 내려갈 수록 너비와 높이 값이 큰 Bounding box이며 하나의 Bounding box에 여러개의 box가 합쳐진 box이다. 여러개의 오브젝트가 있을 확률이 크다.

```python
# rect정보(x1 y1 좌상단 width hight) 만 출력해서 보기
cand_rects = [box['rect'] for box in regions]
print(cand_rects)
```

**bounding box를 시각화 하기**

```python
# opencv의 rectangle()을 이용하여 시각화 그림에 사각형을 그리기
# rectangle()은 이미지와 좌상단 좌표, 우하단 좌표, box컬러색, 두께등을 인자로 입력하면 원본 이미지에 box를 그려줌.

green_rgb = (125, 255, 51)
img_rgb_copy = img_rgb.copy()
for rect in cand_rects:

    left = rect[0]
    top = rect[1]
    # rect[2], rect[3]은 너비와 높이이므로 우하단 좌표를 구하기 위해 좌상단 좌표에 각각을 더함.
    right = left + rect[2]
    bottom = top + rect[3]

    img_rgb_copy = cv2.rectangle(img_rgb_copy, (left, top), (right, bottom), color=green_rgb, thickness=2)
    # 상자를 추가한 Image로 변수 변경

plt.figure(figsize=(8, 8))
plt.imshow(img_rgb_copy)
plt.show()
```

![image](https://user-images.githubusercontent.com/46951365/91564694-f6dab900-e97b-11ea-857d-f3ad6ce70f4d.png)

- bounding box의 크기가 큰 후보만 추출
    - 바로 위에 코드랑 똑같은 코드지만 size만 조금 더 고려

```python
cand_rects = [cand['rect'] for cand in regions if cand['size'] > 10000]

green_rgb = (125, 255, 51)
img_rgb_copy = img_rgb.copy()
for rect in cand_rects:

    left = rect[0]
    top = rect[1]
    # rect[2], rect[3]은 너비와 높이이므로 우하단 좌표를 구하기 위해 좌상단 좌표에 각각을 더함.
    right = left + rect[2]
    bottom = top + rect[3]

    img_rgb_copy = cv2.rectangle(img_rgb_copy, (left, top), (right, bottom), color=green_rgb, thickness=2)

plt.figure(figsize=(8, 8))
plt.imshow(img_rgb_copy)
plt.show()
```

![image](https://user-images.githubusercontent.com/46951365/91564604-cf83ec00-e97b-11ea-9774-6462da33f524.png)

## 2. IOU 적용해보기

- **IOU 구하기**

입력인자로 후보 박스와 실제 박스를 받아서 IOU를 계산하는 함수 생성

```python
import numpy as np

# input에는 (x1 y1 x2 x2) 이미지의 좌상단, 우하단 좌표가 들어가 있다.
def compute_iou(cand_box, gt_box):

    # Calculate intersection areas
    x1 = np.maximum(cand_box[0], gt_box[0])
    y1 = np.maximum(cand_box[1], gt_box[1])
    x2 = np.minimum(cand_box[2], gt_box[2])
    y2 = np.minimum(cand_box[3], gt_box[3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0) # 혹시 모르게 음수가 나올 수 있으니까..

    cand_box_area = (cand_box[2] - cand_box[0]) * (cand_box[3] - cand_box[1])
    gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = cand_box_area + gt_box_area - intersection

    iou = intersection / union
    return iou
```

```python
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

# 실제 box(Ground Truth)의 좌표를 아래와 같다고 가정.
gt_box = [60, 15, 320, 420]

img = cv2.imread('../../data/image/audrey01.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

red = (255, 0 , 0)
img_rgb = cv2.rectangle(img_rgb, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), color=red, thickness=2)

plt.figure(figsize=(8, 8))
plt.imshow(img_rgb)
plt.show()
```

```python
import selectivesearch

#selectivesearch.selective_search()는 이미지의 Region Proposal정보를 반환
_, regions = selectivesearch.selective_search(img_rgb, scale=100, min_size=2000)

print(type(regions), len(regions))
```

    <class 'list'> 53

```python
cand_rects = [cand['rect'] for cand in regions if cand['size'] > 15000]
# cand_box 값도 (좌상단 x1, y1, width, hight) 를 (좌상단, 우하단)의 좌표로 바꾼다.
for index, cand_box in enumerate(cand_rects):
    cand_box = list(cand_box)
    cand_box[2] += cand_box[0]
    cand_box[3] += cand_box[1]

    # 각각의 Box 별로 IOU값을 구해본다
    iou = compute_iou(cand_box, gt_box)
    print('index:', index, "iou:", iou)
```

    index: 0 iou: 0.5933903133903133
    index: 1 iou: 0.20454890788224123
    index: 2 iou: 0.5958024691358025
    index: 3 iou: 0.5958024691358025
    index: 4 iou: 0.1134453781512605
    index: 5 iou: 0.354069104098905
    index: 6 iou: 0.1134453781512605
    index: 7 iou: 0.3278419532685744
    index: 8 iou: 0.3837088388214905
    index: 9 iou: 0.3956795484151107
    index: 10 iou: 0.5008648690956052
    index: 11 iou: 0.7389566501483806
    index: 12 iou: 0.815085997397344
    index: 13 iou: 0.6270619201314865
    index: 14 iou: 0.6270619201314865
    index: 15 iou: 0.6270619201314865

- 바로 위의 코드를 적용해서 이미지와 IOU를 표현하는 이미지를 그려보자.

```python
img = cv2.imread('../../data/image/audrey01.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print('img shape:', img.shape)

green_rgb = (125, 255, 51)
cand_rects = [cand['rect'] for cand in regions if cand['size'] > 3000]
gt_box = [60, 15, 320, 420]
img_rgb = cv2.rectangle(img_rgb, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), color=red, thickness=2)

for index, cand_box in enumerate(cand_rects):

    cand_box = list(cand_box)
    cand_box[2] += cand_box[0]
    cand_box[3] += cand_box[1]

    iou = compute_iou(cand_box, gt_box)

    if iou > 0.7:
        print('index:', index, "iou:", iou, 'rectangle:',(cand_box[0], cand_box[1], cand_box[2], cand_box[3]) )
        cv2.rectangle(img_rgb, (cand_box[0], cand_box[1]), (cand_box[2], cand_box[3]), color=green_rgb, thickness=1)
        text = "{}: {:.2f}".format(index, iou)
        cv2.putText(img_rgb, text, (cand_box[0]+ 100, cand_box[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color=green_rgb, thickness=1)

plt.figure(figsize=(12, 12))
plt.imshow(img_rgb)
plt.show()

```

    img shape: (450, 375, 3)
    index: 3 iou: 0.9874899187876287 rectangle: (59, 14, 321, 421)
    index: 4 iou: 0.9748907882241216 rectangle: (62, 17, 318, 418)
    index: 43 iou: 0.7389566501483806 rectangle: (63, 0, 374, 449)
    index: 44 iou: 0.815085997397344 rectangle: (16, 0, 318, 418)

<img src="https://user-images.githubusercontent.com/46951365/91566429-a1ec7200-e97e-11ea-9b3d-ed5458d6e543.png" style="zoom:80%;" />
