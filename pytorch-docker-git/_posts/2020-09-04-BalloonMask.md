---
layout: post
title: 【Keras】Keras기반 Mask-RCNN - Balloon 데이터셋, Python 추가 공부
description: >
    Keras 기반 Mask-RCNN를 이용해 Balloon 데이터셋을 학습시키고 추론해보자.
---
Keras기반 Mask-RCNN - Balloon 데이터셋  

# 1. 앞으로의 과정 핵심 요약

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/93461157-5135bc80-f91f-11ea-9d82-fec2a49b5286.png' alt='drawing' width='500'/></p>

1. [/Mask_RCNN/tree/master/samples/balloon](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon) 에 있는 내용들을 활용하자. 

2. Matterport Mask RCNN Train 프로세스 및 앞으로의 과정 핵심 정리  
    <p align="center"><img src='https://user-images.githubusercontent.com/46951365/93462114-c35ad100-f920-11ea-88e6-8249dac2ffab.png' alt='drawing' width='700'/></p>  
    <p align="center"><img src='https://user-images.githubusercontent.com/46951365/93462130-c9e94880-f920-11ea-94d2-1dbe199288e5.png' alt='drawing' width='700'/></p>  

# 2. 스스로 공부해보기
- **강의를 보니, 큰 그림은 잡아주신다. 하지만 진짜 공부는 내가 해야한다. 코드를 하나하나 파는 시간을 가져보자. 데이터 전처리 작업으로써 언제 어디서 나중에 다시 사용할 능력일지 모르니, 직접 공부하는게 낫겠다.** 
    - 코드 보는 순서 기록
        1. /mask_rcnn/Balloon_데이터세트_학습및_Segmentation.ipynb
        2. /sample/balloon/balloon.py
        3. /mrcnn/utils.py

# 3. **Python 새로운 핵심 정리**  
python 새롭게 안 사실 및 핵심 내용들
1. super() : [참고 사이트](https://rednooby.tistory.com/56)  
    - \_\_init\_\_ 나 다른 맴버 함수를 포함해서, 자식 클래스에서 아무런 def을 하지 않으면 고대로~ 부모 클래스의 내용이 상속된다. 자식 클래스에서도 함수든 변수든 모두 사용 가능하다.   
    - 하지만 문제가 언제 발생하냐면, def 하는 순간 발생한다. 만약 def \_\_init\_\_(self, ..): 하는 순간, 오버라이딩이 되어 원래 부모 클래스의 내용은 전부 사라지게 된다. 이럴 떄, 사용하는게 super이다.   
    - 대신 클래스 변수를 사용하는 공간에는 super를 사용하지 않아도 상속한 부모 클래스의 내용이 전부 알아서 들어간다. 
    - **super자리에 코드들이 쫘르르륵 들어간다고 생각하라.(마치 해더파일에 있는 함수의 내용이 링크에 의해서 쫘르르 코드가 옮겨 들어가듯)**
    - 아래와 같이, super(MethodFunctionName, self).__init__(객체 생성자에 들어가야 할 input parameter) 를 사용하면, 이렇게 생성된 부모클래스로 만들어진 객체의 맴버변수, 맴버함수를 그대로 사용할 수 있다. 
        ```python
            super(MaskRCNNPredictor, self).__init__(OrderedDict([
                ("conv5_mask", misc_nn_ops.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
                ("relu", nn.ReLU(inplace=True)),
                ("mask_fcn_logits", misc_nn_ops.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
            ]))
        ```
2. VS code - [새롭게 파일을 열 떄 강제로 새로운 탭으로 나오게하는 방법](https://stackoverflow.com/questions/38713405open-files-always-in-a-new-tab) : setting -> workbencheditor.enablePreview" -> false 체크
3. jupyter Notebook font size 바꾸기 : setting -> Editor:Font Size Controls the font size in pixels. -> 원하는size대입
4. **함수안에 함수를 정의하는 행동은 왜 하는 것일까?**   
    그것은 아래와 같은 상황에 사용한다. 함수안의 함수(fun2)는 fun1의 변수를 전역변수처럼 이용할 수 있다. 다시 말해 fun2는 a와 b를 매개변수로 받지 않았지만, 함수 안에서 a와 b를 사용하는 것을 확인할 수 있다.   
    ```python
    def fun1(self, a,b):
        a = 1
        b = 2
        def fun2(c):
            return a + b + c
        k = fun2(3)
        return k
    ```  
    
5. numpy array에 대한 고찰  
    ```python
    import numpy as np
    a = np.ones((10, 10, 3))
    print(np.sum(a, axis=-1).shape)             
    # (10,10) 그냥 2차원 배열이다.
    print(np.sum(a, -1, keepdims=True).shape)   
    np.sum(a, -1, keepdims=True)[:,:,0] 
    # (10,10,1) 3차원 배열의 가장 첫번쨰 원소에 2차원 배열이 들어가 있고, 그 2차원 배열 shape가 10*10 이다.
    ```  
    여기서 주요 요점!   
    - **n차원의 k번째 원소에는(하나의 원소는) n-1차원의배열이다.**
    - 3차원의 1번쨰 원소에는 2차원 배열이다.
    - 2차원의 1번째 원소에는 1차원 배열이다.
    - 1차원의 1번째 원소에는 0차원 배열(스칼라)이다.

6. numpy indexing (인덱싱) 언젠간 공부해야한다. 
    - [https://numpy.org/doc/stable/user/basics.indexing.html](https://numpy.org/doc/stable/user/basics.indexing.html)
7. cv2.circle : 이미지에 점 찍기    
8. skimage.draw.polygon : polygon 정보로 내부 채우기  
9. pretrained weights를 load 할 때, class의 갯수가 다르면head 부분의 layer는 exclude하여 가져오기.

---
     
# 4. 전체 코드 공부해보기

## 1. Balloon data data 준비

- /DLCV/Segmentation/mask_rcnn/Balloon_데이터세트_학습및_Segmentation.ipynb 파일 참조
- Matterport 패키지를 이용하여 Balloon 데이터 세트를 학습하고 이를 기반으로 Segmentation 적용
1. [realse](https://github.com/matterport/Mask_RCNN/releases)에 자료 다운 받기. balloon_dataset.zip 파일   
<p align="center"><img src='https://user-images.githubusercontent.com/46951365/93463397-aaebb600-f922-11ea-856d-ce703b42373e.png' alt='drawing' width='700'/></p>  
2. \~/DLCV/data 에 wget해서 바로 unzip
3. train, val 폴더가 있고, 그 안에 많은 이미지와 json파일 하나 있다.



```python
import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import cv2

%matplotlib inline
```


```python
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
```

- 아래과정 설명
    - [/Mask_RCNN/tree/master/samples/balloon](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon)를 그냥 패키지로 append해버리자.
    - 그리고 그 안에 ballon.py를 쉽게 사용할 수 있다. 
    - [/Mask_RCNN/blob/master/samples/balloon/balloon.py](https://github.com/matterport/Mask_RCNN/blob/master/samples/balloon/balloon.py)파일을 하나하나 뜯어가며 공부해도 좋다. 그대로 사용하는 것보다 공부를 하자!! 
    - 코드를 보면, 아래처럼 구현되어 있는 것을 확인할 수 있다. 우리도 이렇게 해야한다. 
        ```python
        # utils.Dataset을 상속해서 오버로딩을 해야한다. 
        class BalloonDataset(utils.Dataset):
            def load_mask(self, image_id):
        ```  


```python
#Mask_RCNN 패키지의 samples/balloon 디렉토리의 balloon.py 를 import 한다. 
ROOT_DIR = os.path.abspath(".")
sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN/samples/balloon/"))

import balloon
```

- balloon 데이터 세트가 제대로 되어 있는지 확인.  train과 val 서브 디렉토리가 ~/DLCV/data/balloonn 에 존재해야 함. 


```python
import subprocess
from pathlib import Path

HOME_DIR = str(Path.home())
BALLOON_DATA_DIR = os.path.join(HOME_DIR, "DLCV/data/balloon")
```

- balloon 모듈에 설정된 Config 셋업. GPU 갯수, Batch시 image갯수가 사전 설정 되어 있음. 
- BalloonConfig도 mrcnn.config를 상속해서 사용하는 것을 알 수 있다.


```python
config = balloon.BalloonConfig()
config.IMAGES_PER_GPU = 1
config.display()
```

    
    Configurations:
    BACKBONE                       resnet101
    BACKBONE_STRIDES               [4, 8, 16, 32, 64]
    BATCH_SIZE                     2
    BBOX_STD_DEV                   [0.1 0.1 0.2 0.2]
    COMPUTE_BACKBONE_SHAPE         None
    ...
    USE_MINI_MASK                  True
    USE_RPN_ROIS                   True
    VALIDATION_STEPS               50
    WEIGHT_DECAY                   0.0001
    
    


- balloon 모듈에서 balloon 데이터 세트 로딩. 


```python
# Dataset 로딩한다. . 

dataset = balloon.BalloonDataset()
dataset.load_balloon(BALLOON_DATA_DIR, "train")

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))
```

    Image Count: 61
    Class Count: 2
      0. BG                                                
      1. balloon                                           


- balloon 모듈에서 로딩한 balloon 데이터 세트의 세부 정보 확인. 


```python
# dataset의 image_info는 리스트 객체이며 내부 원소로 이미지별 세부 정보를 딕셔너리로 가지고 있음. 
# dataset의 image_ids 는 이미지의 고유 id나 이름이 아니라 dataset에서 이미지의 상세 정보를 관리하기 위한 리스트 인덱스에 불과 

print('#### balloon 데이터 세트 이미지의 인덱스 ID들 ####')
print(dataset.image_ids)
# print('\n ##### balloon 데이터 세트의 이미지 정보들 ####')
# print(dataset.image_info)
```

    #### balloon 데이터 세트 이미지의 인덱스 ID들 ####
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
     24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
     48 49 50 51 52 53 54 55 56 57 58 59 60]


## 2.  가져온 데이터 Load & Read 해보기
- balloon.py 파일의 BalloonDataset.Load_mask 함수를 보면, polygon정보를 True-False mask 정보로 바꾸는 것을 확인할 수 있다. 


```python
image_28 = dataset.image_info[28]  # 28 index 이미지의 정보를 가져온다
print(image_28,'\n')
polygons = image_28['polygons']
polygon_x = polygons[0]['all_points_x']
polygon_y = polygons[0]['all_points_y']
print(len(polygon_x))
# print('polygon_x:', polygon_x, 'polygon_y:',polygon_y)

polygon_xy = [(x, y) for (x, y) in zip(polygon_x, polygon_y)]
print('polygon_xy:', polygon_xy)
```

    {'id': '5178670692_63a4365c9c_b.jpg', 'source': 'balloon', 'path': '/home/sb020518/DLCV/data/balloon/train/5178670692_63a4365c9c_b.jpg', 'width': 683, 'height': 1024, 'polygons': [{'name': 'polygon', 'all_points_x': [371, 389, 399, 409, 416, 415, 407, 395, 381, 364, 346, 331, 327, 324, 322, 318, 316, 319, 304, 290, 276, 280, 289, 304, 326, 351, 371], 'all_points_y': [424, 432, 443, 459, 482, 505, 526, 544, 558, 567, 573, 576, 576, 580, 580, 577, 574, 572, 562, 543, 509, 477, 451, 436, 423, 420, 424]}]} 
    
    27
    polygon_xy: [(371, 424), (389, 432), (399, 443), (409, 459), (416, 482), (415, 505), (407, 526), (395, 544), (381, 558), (364, 567), (346, 573), (331, 576), (327, 576), (324, 580), (322, 580), (318, 577), (316, 574), (319, 572), (304, 562), (290, 543), (276, 509), (280, 477), (289, 451), (304, 436), (326, 423), (351, 420), (371, 424)]



```python
image_28_array = cv2.imread(os.path.join(BALLOON_DATA_DIR,'train/'+image_28['id']))
for position in polygon_xy:
    cv2.circle(image_28_array, position, 3, (255, 0, 0), -1) # 이미지에 점을 찍는 함수

# plt.figure(figsize=(8, 8))
# plt.axis('off')    
# plt.imshow(image_28_array)
```

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/94019976-539b8900-fded-11ea-8c35-01a77ba4fa3b.png' alt='drawing' width='300'/></p>


```python
np.random.seed(99)
# Load and display random samples
image_ids = np.random.choice(dataset.image_ids, 4)
print('image_ids:', image_ids)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    # 지정된 image_id에 있는 mask 를 로딩하고 시각화를 위한 mask정보들과 대상 클래스 ID들을 추출
    mask, class_ids = dataset.load_mask(image_id)
    #원본 데이터와 여러개의 클래스들에 대해 Mask를 시각화 하되, 가장 top 클래스에 대해서는 클래스명까지 추출. 나머지는 배경
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)
```

    image_ids: [ 1 35 57 40]


<p align="center"><img src='https://user-images.githubusercontent.com/46951365/94020753-3dda9380-fdee-11ea-8ea7-77417383f713.png' alt='drawing' width='400'/></p>


```python
image = dataset.load_image(28)
print(image.shape)
print(image_28['polygons'])
```

## 3. polygon 형태의 데이터를 boolean mask 형태로 변환
- balloon.py의 dataset.load_mask 함수 내용 참조
- skimage.draw.polygon이라는 함수 적극적으로 사용


```python
import skimage

img = np.zeros((10, 10), dtype=np.uint8)
r = np.array([1, 2, 8])
c = np.array([1, 7, 4])
plt.imshow(img, cmap='gray')
plt.show()
# r과 c로 지정된 인덱스에 있는 img 값만 1로 설정함. 
rr, cc = skimage.draw.polygon(r, c)
img[rr, cc] = 1
print('row positions:',rr, 'column positions:',cc)
print('Boolean형태로 masking된 img:\n',img.astype(np.bool))
plt.imshow(img, cmap='gray')
```

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/94022587-36b48500-fdf0-11ea-854b-bcceb11514cd.png' alt='drawing' width='300'/></p>


```python
mask, class_ids = dataset.load_mask(59)
print("mask shape:", mask.shape, "class_ids:", class_ids)
# print(mask)
```

    mask shape: (679, 1024, 28) class_ids: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]



```python
image = dataset.load_image(28)
mask, class_ids = dataset.load_mask(28)
visualize.display_top_masks(image, mask, class_ids, dataset.class_names)
```

## 4. ballon 데이터 세트를 다루기 위한 BallonDataset Class 정의
- 아래의 코드는 ballon.py의 class BalloonDataset(utils.Dataset): 내용과 동일하다. 여기서 보면 load_balloon를 사용해서 우리가 가진 이미지에 대해서 정의한다. 단, json파일은 이미 만들어져 있어야 한다. 
- 그리고 위의 코드에서 dataset.load_balloon(BALLOON_DATA_DIR, "train");  dataset.prepare(); 을 사용했던 것을 확인할 수 있다.


```python
class BalloonDataset(utils.Dataset):

    def load_balloon(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # 클래스 id와 클래스명 등록은 Dataset의 add_class()를 이용. 
        self.add_class("balloon", 1, "balloon")

        # train또는 val 용도의 Dataset 생성만 가능. 
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        # json 형태의 annotation을 로드하고 파싱. 
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys
        
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "balloon",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "balloon":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
    
    '''def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
    '''
```

## 5. balloon 데이터 세트의 학습 수행. 
- 학습과 Validation용 Dataset 설정.


```python
import skimage

# Training dataset.
dataset_train = BalloonDataset()
dataset_train.load_balloon(BALLOON_DATA_DIR, "train")
dataset_train.prepare()

# Validation dataset
dataset_val = BalloonDataset()
dataset_val.load_balloon(BALLOON_DATA_DIR, "val")
dataset_val.prepare()
```

- Config 설정 : 이것도 balloon.py에 있는 내용을 거의 그대로!


```python
from mrcnn.config import Config

TRAIN_IMAGE_CNT = len(dataset_train.image_info)
VALID_IMAGE_CNT = len(dataset_val.image_info)

class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "balloon"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    
    # 추가.
    GPU_COUNT = 1

    # 원본에서 수정.
    #STEPS_PER_EPOCH = TRAIN_IMAGE_CNT  // IMAGES_PER_GPU
    #VALIDATION_STEPS = VALID_IMAGE_CNT  // IMAGES_PER_GPU
    
    # 원본 STEPS_PER_EPOCH
    STEPS_PER_EPOCH = TRAIN_IMAGE_CNT  // IMAGES_PER_GPU
    VALIDATION_STEPS = VALID_IMAGE_CNT  // IMAGES_PER_GPU

    #BACKBONE = 'resnet101'
    
# config 설정. 
train_config = BalloonConfig()
train_config.display()
```

## 6. Mask RCNN Training 초기 모델 생성 및 pretrained weight값 로딩
- 여기서 부터는 (/mrcnn/model.py 내부에)karas의 내용들이 다수 들어가기 때문에 어려울 수 있으니 참고.


```python
import mrcnn.model as modellib
from mrcnn.model import log

balloon_model = modellib.MaskRCNN(mode="training", config=train_config, model_dir='./snapshots')

# COCO 데이터 세트로 pretrained 된 모델을 이용하여 초기 weight값 로딩. 
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "./pretrained/mask_rcnn_coco.h5")
balloon_model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
```

## 7. 위에서 가져온 pretrained model을 가지고 학습 수행


```python
'''
데이터 세트가 작고,단 하나의 클래스임. 
pretrained 된 Coco 데이터 세트로 초기 weight 설정되었기에 RPN과 classifier만 학습해도 모델 성능은 큰 영향이 없을 거라 예상
all: All the layers
3+: Train Resnet stage 3 and up
4+: Train Resnet stage 4 and up
5+: Train Resnet stage 5 and up
'''
print("Training network heads")
balloon_model.train(dataset_train, dataset_val,
            learning_rate=train_config.LEARNING_RATE,
            epochs=30,
            layers='heads')
```

    Training network heads
    
    Starting at epoch 0. LR=0.001
    
    Checkpoint Path: ./snapshots/balloon20200923T1430/mask_rcnn_balloon_{epoch:04d}.h5
    Selecting layers to train
    fpn_c5p5               (Conv2D)
    fpn_c4p4               (Conv2D)
    fpn_c3p3               (Conv2D)
    fpn_c2p2               (Conv2D)
    fpn_p5                 (Conv2D)
    fpn_p2                 (Conv2D)
    fpn_p3                 (Conv2D)
    fpn_p4                 (Conv2D)
    In model:  rpn_model
        rpn_conv_shared        (Conv2D)
        rpn_class_raw          (Conv2D)
        rpn_bbox_pred          (Conv2D)
    mrcnn_mask_conv1       (TimeDistributed)
    mrcnn_mask_bn1         (TimeDistributed)
    mrcnn_mask_conv2       (TimeDistributed)
    mrcnn_mask_bn2         (TimeDistributed)
    mrcnn_class_conv1      (TimeDistributed)
    mrcnn_class_bn1        (TimeDistributed)
    mrcnn_mask_conv3       (TimeDistributed)
    mrcnn_mask_bn3         (TimeDistributed)
    mrcnn_class_conv2      (TimeDistributed)
    mrcnn_class_bn2        (TimeDistributed)
    mrcnn_mask_conv4       (TimeDistributed)
    mrcnn_mask_bn4         (TimeDistributed)
    mrcnn_bbox_fc          (TimeDistributed)
    mrcnn_mask_deconv      (TimeDistributed)
    mrcnn_class_logits     (TimeDistributed)
    mrcnn_mask             (TimeDistributed)
    
    Use tf.cast instead.
    Epoch 1/30
    Process Process-2:
    Process Process-3:
    Process Process-5:
    Process Process-6:
    
    


## 8. 학습이 완료된 모델을 이용하여 inference 수행. 
- config를 inference용으로 변경


```python
class InferenceConfig(BalloonConfig):
    # NAME은 학습모델과 동일한 명을 부여
    NAME='balloon'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
        
infer_config = InferenceConfig()
infer_config.display()
```

- 학습된 모델의 weight 파일을 MaskRCNN의 inference 모델로 로딩. 


```python
model = modellib.MaskRCNN(mode="inference", model_dir='./snapshots', config=infer_config)
# callback에 의해 model weights 가 파일로 생성되며, 가장 마지막에 생성된 weights 가 가장 적은 loss를 가지는 것으로 가정. 
weights_path = model.find_last()
print('최저 loss를 가지는 model weight path:', weights_path)
# 지정된 weight 파일명으로 모델에 로딩. 
model.load_weights(weights_path, by_name=True)
```

- Instance Segmentation을 수행할 파일들을 dataset로 로딩. val 디렉토리에 있는 파일들을 로딩. 


```python
# Inference를 위해 val Dataset 재로딩. 
dataset_val = BalloonDataset()
dataset_val.load_balloon(BALLOON_DATA_DIR, "val")
dataset_val.prepare()

print("Images: {}\nClasses: {}".format(len(dataset_val.image_ids), dataset_val.class_names))
```


```python
- 아래처럼 model.detect을 사용해서 아주 쉽게, Detect 결과 추출!
```


```python
from mrcnn import model as modellib

# dataset중에 임의의 파일을 한개 선택. 
#image_id = np.random.choice(dataset.image_ids)
image_id = 5
image, image_meta, gt_class_id, gt_bbox, gt_mask=modellib.load_image_gt(dataset_val, infer_config, image_id, use_mini_mask=False)
info = dataset_val.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset_val.image_reference(image_id)))

# Run object detection
results = model.detect([image], verbose=1)
```

![image](https://user-images.githubusercontent.com/46951365/94268836-b4a39800-ff78-11ea-8d7a-2d5559a40b4a.png)


```python
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], 
                            title="Predictions")
```

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/94268888-d1d86680-ff78-11ea-8972-1d8830107298.png' alt='drawing' width='500'/></p>

## 9. Detect 결과에서 풍선만 칼라로. 나머지는 흑백으로 바꾸자.


```python
#Mask_RCNN 패키지의 samples/balloon 디렉토리의 balloon.py 를 import 한다. 
ROOT_DIR = os.path.abspath(".")
sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN/samples/balloon/"))

import balloon
from mrcnn.visualize import display_images

splash = balloon.color_splash(image, r['masks'])
display_images([splash], cols=1)
```

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/94269234-57f4ad00-ff79-11ea-9f96-860dbb171bf5.png' alt='drawing' width='500'/></p>

- balloon.py 내부의 color_splash 함수는 다음과 같다.   


```python
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image. (풍선을 제외하고 Gray로 변환하기)
    """
    # '칼라->흑백->칼라' 과정을 거쳐서, gray 값을 3 depth로 가지는 3채널 이미지 만듬
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1) # 전체 풍선에 대해서, 풍선이 위치했던 곳에 대한 Mask 정보(True&False) 추출
        # An array with elements from x where condition is True, and elements from y elsewhere.
        # https://numpy.org/doc/stable/reference/generated/numpy.where.html
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash
```

- 어떻게 위와 같은 함수가 동작할까?


```python
print('image shape:',image.shape, 'r mask shape:',r['masks'].shape)
mask = (np.sum(r['masks'], -1, keepdims=True) >= 1)
print('sum mask shape:',mask.shape)
```

- np.sum() 테스트


```python
import numpy as np
a = np.ones((10, 10, 3))
#print(a)
#print(np.sum(a))
print(np.sum(a, axis=-1).shape)             # 그냥 2차원 배열이다.
print(np.sum(a, -1, keepdims=True).shape)   # 3차원 배열의 가장 첫번쨰 원소에 2차원 배열이 들어가 있고, 그 2차원 배열 shape가 10*10 이다.
print(np.sum(a, -1, keepdims=True) >=1 )
print(np.sum(a, -1, keepdims=True)[:,:,0])

```

    (10, 10)
    (10, 10, 1)


- np.where() 테스트


```python
test_mask = (np.sum(a, -1, keepdims=True) >=1)
print(test_mask.shape)
for i in range(5):
    for j in range(5):
        test_mask[i, j, 0] = False
        
test_image = np.ones((10, 10, 3))
test_gray = np.zeros((10, 10, 3))
np.where(test_mask, test_image, test_gray)[:,:,0]
```

    (10, 10, 1)





    array([[0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
           [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
           [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
           [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
           [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])



## 10. Video에 Inferece 적용 후 color splash를 적용. 


```python
from IPython.display import clear_output, Image, display, Video, HTML
Video('../../data/video/balloon_dog02.mp4')
```

- Video color splash를 적용한 함수를 생성해보자. 
- 이전부터 사용했던, cv2.VideoCapture 그대로를 사용할 것이다!


```python
import cv2
import time

def detect_video_color_splash(model, video_input_path=None, video_output_path=None):

    cap = cv2.VideoCapture(video_input_path)
    codec = cv2.VideoWriter_fourcc(*'XVID')
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    vid_writer = cv2.VideoWriter(video_output_path, codec, fps, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                                 round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("총 Frame 개수: {0:}".format(total))

    frame_index = 0
    success = True
    while True:
        
        hasFrame, image_frame = cap.read()
        if not hasFrame:
            print('End of frame')
            break
        frame_index += 1
        print("frame index:{0:}".format(frame_index), end=" ")
        
        # OpenCV returns images as BGR, convert to RGB
        image_frame = image_frame[..., ::-1] 
        # ::-1은 slice step. 시작은 무조건 0 끝은 len
        print('End of frame')
            break
        frame_index += 1
        print("frame index:{0:}".format(frame_index), end=" ")
        
        # OpenCV returns images as BGR, convert to RGB
        image_frame = image_frame[..., ::-1] 
        # ::-1은 slice step. 시작은 무조건 0 끝은 len
        # https://numpy.org/doc/stable/user/basics.indexing.html#combining-index-arrays-with-slices 참조
        start=time.time()
        # Detect objects
        r = model.detect([image_frame], verbose=0)[0]
        print('detected time:', time.time()-start)
        # Color splash
        splash = color_splash(image_frame, r['masks'])
        # RGB -> BGR to save image to video
        splash = splash[..., ::-1] 
        # Add image to video writer
        vid_writer.write(splash)
    
    vid_writer.release()
    cap.release()       
    
    print("Saved to ", video_output_path)
    
detect_video_color_splash(model, video_input_path='../../data/video/balloon_dog02.mp4', 
                          video_output_path='../../data/output/balloon_dog02_output.avi')

```


      File "<ipython-input-20-f12d9e351b0b>", line 30
        break
        ^
    IndentationError: unexpected indent



- numpy index 머리가 너무 아프지만, 알아두면 분명히 좋을 것이다. 일단 필수 사이트는 아래와 같다
- 참조사이트 : [https://numpy.org/doc/stable/user/basics.indexing.html](https://numpy.org/doc/stable/user/basics.indexing.html)

    ```python
    >>> y
    array([[ 0,  1,  2,  3,  4,  5,  6],
           [ 7,  8,  9, 10, 11, 12, 13],
           [14, 15, 16, 17, 18, 19, 20],
           [21, 22, 23, 24, 25, 26, 27],
           [28, 29, 30, 31, 32, 33, 34]])
    >>> y[0][1]
    1
    >>> y[0,1]
    1
    >>> y[1]
    array([ 7,  8,  9, 10, 11, 12, 13])
    >>> y[3]
    array([21, 22, 23, 24, 25, 26, 27])
    >>> y[[1,3]]
    array([[ 7,  8,  9, 10, 11, 12, 13],
           [21, 22, 23, 24, 25, 26, 27]])
    >>> y[1:5:2]
    array([[ 7,  8,  9, 10, 11, 12, 13],
           [21, 22, 23, 24, 25, 26, 27]])
    >>> y[1:5:2,[0,3,6]]
    array([[ 7, 10, 13],
           [21, 24, 27]])
    >>> y[1:5:2,::3]
    array([[ 7, 10, 13],
           [21, 24, 27]])

    >>> z[1,...,2]
    array([[29, 32, 35],
           [38, 41, 44],
           [47, 50, 53]])
    >>> z[1,:,:,2]
    array([[29, 32, 35],
           [38, 41, 44],
           [47, 50, 53]])
    >>> z[1] == z[1,:,:,:]
    True
    >>> z[1,...,2] == z[1,:,:,2]
    True # 즉( :,:,:,: 을 줄여쓰고 싶면 ... 을 사용하면 된다.)
    ```       

- 생성된 Output 파일을 Object Storage에 저장한 뒤 확인


```python
!gsutil cp ../../data/output/balloon_dog02_output.avi gs://my_bucket_dlcv/data/output/balloon_dog02_output.avi
```

# 5. 데이터 학습을 위한 전체 흐름도   
<p align="center"><img src='https://user-images.githubusercontent.com/46951365/94285948-5b932e80-ff8f-11ea-847a-6d9427790cf3.png' alt='drawing' width='700'/></p>