---
layout: post
title: 【Keras】Keras기반 Mask-RCNN - Kaggle Necleus 데이터셋
description: >
    Keras 기반 Mask-RCNN를 이용해 Kaggle Necleus 데이터셋을 학습시키고 추론해보자.
---
【Keras】Keras기반 Mask-RCNN - Kaggle Necleus 데이터셋

# 1. python 핵심 정리 모음(새롭게 안 사실)  
python 새롭게 안 사실 및 핵심 내용들  
1. lines tab 처리하는 방법  
    - ctrl + \[
    - ctrl + \]
    - line drag + tab
    - line drag + shift + tab
2. os package 유용한 함수들
    - os.listdir : 폴더 내부 파일이름들 list로 따옴
    - os.walk : sub directory를 iteration으로 반환 (검색해서 아래 활용 예시 보기)
    - [os.path.dirname](http://pythonstudy.xyz/python/article/507-%ED%8C%8C%EC%9D%BC%EA%B3%BC-%EB%94%94%EB%A0%89%ED%86%A0%EB%A6%AC) : 경로 중 이전 dir에 대한 정보까지만 얻기
3. image_ids = list(set(image_ids) - set(val_indexs)) 
    - set과 list를 동시에 이용한다. 교집합을 제거하고, 남은 내용들만 list로 반환
4. 문자열함수 [endwith(문자열)](http://www.w3big.com/ko/python/att-string-endswith.html) : 문자열에 해당 문자가 있으면 True, 없으면 False
5. skimage.io.imread("bool 형식의 mask data 파일 이름").astype(np.bool) 
    - mask 데이터 뽑아 오는 하나의 방법
6. masks = np.stack(mask_list, axis=-1) 
    - 한장한장의 mask 데이터를 concatenation하는 방법
    - [공식 dacument](https://numpy.org/doc/stable/reference/generated/numpy.stack.html)
    - 만약 내가 묶고 싶은 list의 원소가 n개 일 때, axis가 명시하는 shape의 index가 n이 된다. 
    - **2**개의 list(1*3)를 합치고 axis가 -1이라면, 최종 result는 3x**2**가 된다. 

---

# 2. Keras기반 Mask-RCNN - Kaggle Necleus 데이터셋 정의하고 학습하기
- /DLCV/Segmentation/mask_rcnn/Kaggle_Nucleus_Segmentation_Challenge.ipynb 참고하기
- 전체 흐름도 정리 : [이전 포스트 자료](https://junha1125.github.io/docker-git-pytorch/2020-09-04-BalloonMask/#4-%EB%8D%B0%EC%9D%B4%ED%84%B0-%ED%95%99%EC%8A%B5%EC%9D%84-%EC%9C%84%ED%95%9C-%EC%A0%84%EC%B2%B4-%ED%9D%90%EB%A6%84%EB%8F%84)

## 1. Kaggle API로 데이터 다운로드 및 데이터 탐방
- 이전에 배웠던 [winSCP](https://junha1125.github.io/ubuntu-python-algorithm/2020-08-05-Google_cloud2/#3-cloud-%EC%82%AC%EC%9A%A9%EC%8B%9C-%EC%A3%BC%EC%9D%98%EC%82%AC%ED%95%AD-%EB%B0%8F-object-storage-%EC%84%A4%EC%A0%95), 서버의 directory 구조 그대로 보고, 데이터를 주고 받을 수 있는 프로그램, 을 사용해도 좋다. 하지만 여기서는 kaggle API를 이용해서 data받는게 빠르고 편하다. 
- kaggle API 설치 및 데이터 다운로드 [사용법 사이트](https://teddylee777.github.io/kaggle/Kaggle-API-%EC%82%AC%EC%9A%A9%EB%B2%95)

```python
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
%matplotlib inline 
```


```python
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
```



```python
## Kaggle에서 2018 Data science bowl Nucleus segmentation 데이터를 download 한 뒤 ./nucleus_data 디렉토리에 압축을 품
# stage1_train.zip 파일에 train용 image 데이터와 mask 데이터가 있음. stage1_train_zip 파일을 stage1_train 디렉토리에 압축 해제
# unzip stage1_train.zip -d stage1_train
# stage1_test.zip 파일에 test용 image 데이터와 mask 데이터가 있음. stage1_test_zip 파일을 stage1_test 디렉토리에 압축 해제
# unzip stage1_test.zip -d stage1_test
```

- **train용 데이터 세트에 들어있는 데이터 구조 확인**


```python
import os
from pathlib import Path

HOME_DIR = str(Path.home())

# 학습용, 테스트용 모두의 기준 디렉토리는 ~/DLCV/data/nucleus 임. 
DATASET_DIR = os.path.join(HOME_DIR, "DLCV/data/nucleus")
# print(DATASET_DIR)
```


```python
# ~/DLCV/data/nucleus 디렉토리 밑에 학습용 디렉토리인 stage1_train이 만들어짐.stage1_train에 학습 이미지, mask 이미지 데이터 존재. 
subset_dir = 'stage1_train'
train_dataset_dir = os.path.join(DATASET_DIR, subset_dir)
# print(train_dataset_dir)
```

- train 데이터 세트의 이미지 파일, mask 파일이 어떠한 디렉토리 구조 형태로 저장되어 있는지 확인. 
- /DLCV/data/nucleus/stage1_train/9a71a416f98971aa14f63ef91242654cc9191a1414ce8bbd38066fe94559aa4f$ ls  
    \>> images  masks
- 아래 **코드 주의 os.walk 사용**하는 중요한 코드!!

```python
# 이미지 별로 고유한 이미지명을 가지는 이미지 디렉토리를 가지고 이 디렉토리에 하위 디렉토리로 images, masks를 가짐
# images 에는 하나의 이미지가 있으며 masks는 여러개의 mask 이미지 파일을 가지고 있음. 즉 하나의 이미지에 여러개의 mask 파일을 가지고 있는 형태임. 
# next(os.walk(directory))[1]은 sub directory를 iteration으로 반환 next(os.walk(directory))[2]는 해당 디렉토리 밑에 파일들을 iteration으로 반환
# 즉 [1]은 directory를 반환하고
#    [2]는 파일을 반환한다. 
index = 0 
for dir in next(os.walk(train_dataset_dir))[1]:
    print('┌',dir)
    subdirs = os.path.join(train_dataset_dir, dir)
    for subdir in next(os.walk(subdirs))[1]:
        print('└┬─'+subdir)
        sub_subdirs = os.path.join(subdirs, subdir)
        for sub_subdir in next(os.walk(sub_subdirs))[2]:
            print(' └── '+sub_subdir) # sub_subdir 에서는 png를 가져온 것을 확인할 수 있다.
            index += 1
            if index >1000:
                break
```

![image](https://user-images.githubusercontent.com/46951365/94460930-67335f00-01f4-11eb-90b7-01d3f7bcd26a.png)

- 하나의 dir에는 '하나의 이미지(images)'와 '그 이미지에 대한 새포핵 하나하나에 대한 mask 흑백이미지들'이 저장되어 있다. 

## 2. Dataset 객체와 Data load 메소드 작성

- 학습시 사용할 임의의 Validation용 IMAGE를 무엇으로 사용할지 랜덤 설정


```python
def get_valid_image_ids(dataset_dir, valid_size):
    np.random.seed(0)
    dataset_dir = os.path.join(dataset_dir,'stage1_train')
    image_ids = next(os.walk(dataset_dir))[1] # stage1_train 딱 내부의 폴더들에 대한 iteration 가져옴. [1]은 '폴더'를 의미. 
    total_cnt = len(image_ids) # stage1_train 딱 내부의 폴더들의 갯수를 가져옴
    
    valid_cnt = int(total_cnt * valid_size) # 0 < valid_size < 1 을 명심하기
    valid_indexes = np.random.choice(total_cnt, valid_cnt, replace=False)
    
    return total_cnt, list(np.array(image_ids)[valid_indexes])  # stage1_train 내부 이미지 중, valid에 사용할 폴더의 index(몇번째 폴더)만 가져온다. 
```


```python
total_cnt, val_indexs = get_valid_image_ids(DATASET_DIR, 0.1) 
print(total_cnt, len(val_indexs))
val_indexs[0:5]
```

    670 67

    ['d7fc0d0a7339211f2433829c6553b762e2b9ef82cfe218d58ecae6643fa8e9c7',
     '6ab24e7e1f6c9fdd371c5edae1bbb20abeeb976811f8ab2375880b4483860f4d',
     '40b00d701695d8ea5d59f95ac39e18004040c96d17fbc1a539317c674eca084b',
     '1ec74a26e772966df764e063f1391109a60d803cff9d15680093641ed691bf72',
     '431b9b0c520a28375b5a0c18d0a5039dd62cbca7c4a0bcc25af3b763d4a81bec']



- utils.Dataset 객체의 add_class(), add_image()를 이용하여, 개별 이미지를 Dataset 객체로 로딩하는 load_nucleus() 함수 정의하기. 


```python
from mrcnn import utils
import skimage
import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

_, val_indexs = get_valid_image_ids(DATASET_DIR, 0.1)

class NucleusDataset(utils.Dataset):
    
    # subset은 train, valid, stage1_test, stage2_test
    def load_nucleus(self, dataset_dir, subset):
        self.add_class(source='nucleus', class_id=1, class_name='nucleus')  
        # 이미 class로 등록되어 있으면, 지금 데이터셋의 class로 새로 넣지 않음
        # 새로운 데이터셋이라는 것을 명칭해주는 것이 source이다. "우리는 nucleus 라는 데이터셋을 추가로 등록할 것이다."를 의미함 
        
        subset_dir = 'stage1_train' if subset in ['train', 'val'] else subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        
        if subset=='val':
            image_ids = val_indexs
        else:
            image_ids = next(os.walk(dataset_dir))[1]  # 폴더들 이름을 전체 list로 저장
            if subset=='train':
                image_ids = list(set(image_ids) - set(val_indexs)) # 

        for image_id in image_ids: # image_id라는 폴더 내부에 images_<image_id>.png 라는 이미지가 있다고 전제
            self.add_image('nucleus', image_id=image_id, path=os.path.join(dataset_dir, image_id, 'images/{}.png'.format(image_id)))   
    
    def load_mask(self, image_id):      ## 오버라이딩으로 정의해주지 않으면 오류나는 함수!  image_id는 숫자
        info = self.image_info[image_id]
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), 'masks')
        mask_list=[]
        for mask_file in next(os.walk(mask_dir))[2]: # mask_dir에 있는 "파일" 이름들을 list로 가지고 있음.
            if mask_file.endswith('.png'):
                mask = skimage.io.imread(os.path.join(mask_dir, mask_file)).astype(np.bool)
                mask_list.append(mask)

                # test_mask = cv2.imread(os.path.join(mask_dir, mask_file))
                # print(type(test_mask), test_mask.shape) -> 하나의 파일은 <class 'numpy.ndarray'> (256, 320, 3)
                
        masks = np.stack(mask_list, axis=-1) 
        
        return masks, np.ones([masks.shape[-1]], dtype=np.int32)
```



```python
nucleus_dataset = NucleusDataset(utils.Dataset)
nucleus_dataset.load_nucleus(DATASET_DIR, 'train')
#print('class info:', nucleus_dataset.class_info)
#print('image info:', nucleus_dataset.image_info)
nucleus_dataset.prepare()
```


```python
print('class id:', nucleus_dataset.class_ids)
print('image id:', nucleus_dataset._image_ids)
```

    class id: [0 1]
    image id: [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
      18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
      36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53
      54  55  56  57  58  59  60  ...  586 587 588 589 590 591 592 593
     594 595 596 597 598 599 600 601 602]



```python
nucleus_dataset.image_info[0]
```

{  
'id': '9f073db4acd7e634fd578af50d4e77218742f63a4d423a99808d6fd7cb0d3cdb',  
'source': 'nucleus',  
'path': '/home/sb020518/DLCV/data/nucleus/stage1_train/9f073db4acd7e634fd578af50d4e77218742f63a4d423a99808d6fd7cb0d3cdb/images9f073db4acd7e634fd578af50d4e77218742f63a4d423a99808d6fd7cb0d3cdb.png'  
}  




```python
masks, class_ids = nucleus_dataset.load_mask(0)
masks.shape, class_ids.shape
```




    ((256, 256, 18), (18,))




```python
masks, class_ids = nucleus_dataset.load_mask(0)

sample_img = skimage.io.imread("/home/sb020518/DLCV/data/nucleus/stage1_train/9f073db4acd7e634fd578af50d4e77218742f63a4d423a99808d6fd7cb0d3cdb/images/9f073db4acd7e634fd578af50d4e77218742f63a4d423a99808d6fd7cb0d3cdb.png")
print(type(sample_img), sample_img.shape)
plt.imshow(sample_img)
plt.show()

print(masks[0]) 
```

    <class 'numpy.ndarray'> (256, 256, 4)

    [[False False False ... False False False]
     [False False False ... False False False]
     [False False False ... False False False]
     ...
     [False False False ... False False False]
     [False False False ... False False False]
     [False False False ... False False False]]


<img src='https://user-images.githubusercontent.com/46951365/94417808-8366da00-01bb-11eb-8ca2-17c6adad7978.png' alt='drawing' width='250'/>


```python
img_from_cv = cv2.imread("/home/sb020518/DLCV/data/nucleus/stage1_train/9f073db4acd7e634fd578af50d4e77218742f63a4d423a99808d6fd7cb0d3cdb/images/9f073db4acd7e634fd578af50d4e77218742f63a4d423a99808d6fd7cb0d3cdb.png")
print(type(img_from_cv), img_from_cv.shape)
plt.imshow(img_from_cv)
plt.show()

```

    <class 'numpy.ndarray'> (256, 256, 3)

<img src='https://user-images.githubusercontent.com/46951365/94417808-8366da00-01bb-11eb-8ca2-17c6adad7978.png' alt='drawing' width='250'/>


```python
np.random.seed(0)
image_ids = np.random.choice(nucleus_dataset.image_ids, 4) # 506개의 이미지 중에 4개를 선택한다. 
print(image_ids)
for image_id in image_ids:
    image = nucleus_dataset.load_image(image_id)
    mask, class_ids = nucleus_dataset.load_mask(image_id)
    print('mask shape:', mask.shape, 'class_ids shape:', class_ids.shape)
    visualize.display_top_masks(image, mask, class_ids, nucleus_dataset.class_names, limit=1)
```

    [559 192 359   9]
    mask shape: (256, 256, 6) class_ids shape: (6,)
    mask shape: (360, 360, 24) class_ids shape: (24,)
    mask shape: (256, 256, 16) class_ids shape: (16,)
    mask shape: (256, 256, 4) class_ids shape: (4,)


<img src='https://user-images.githubusercontent.com/46951365/94418009-c6c14880-01bb-11eb-8878-088587dd928c.png' alt='drawing' width='400'/>

## 3. nucleus 데이터 세트 정의 및 pretrain 가져오기
- Matterport 패키지에서 사용될 수 있도록 학습/검증/테스트 데이터 세트에 대한 각각의 객체 생성 필요!
- 학습 또는 Inference를 위한 Config 설정
- 학습과정 정리  
    <img src='https://user-images.githubusercontent.com/46951365/94418528-7c8c9700-01bc-11eb-81c3-a8fe7937cffd.png' width='500'/>



```python
dataset_train = NucleusDataset()
dataset_train.load_nucleus(DATASET_DIR, 'train')  # 위에서 NucleusDataset의 맴버 함수로 설정했었음
# dataset을 load한 뒤에는 반드시 prepare()메소드를 호출
dataset_train.prepare()

# Validation dataset
dataset_val = NucleusDataset()
dataset_val.load_nucleus(DATASET_DIR, "val")
dataset_val.prepare()
```


```python
len(dataset_train.image_info), len(dataset_val.image_info)
# (603, 67)
```

- Nucleus 학습을 위한 새로운 Config 객체 생성


```python
from mrcnn.config import Config

train_image_cnt = len(dataset_train.image_info)
val_image_cnt = len(dataset_val.image_info)
print('train_image_cnt:',train_image_cnt, 'validation image count:',val_image_cnt)

class NucleusConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "nucleus"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (train_image_cnt) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, (val_image_cnt // IMAGES_PER_GPU))

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400
```

    train_image_cnt: 603 validation image count: 67


- coco pretrained된 가중치 모델 가져오기.
- [과거](https://junha1125.github.io/docker-git-pytorch/2020-09-03-2kerasMask/#1--matterport-%ED%8C%A8%ED%82%A4%EC%A7%80-%EB%AA%A8%EB%93%88-%EB%B0%8F-%ED%81%B4%EB%9E%98%EC%8A%A4-%ED%99%9C%EC%9A%A9)에 아래와 같은 코드로 패키지에 있는 pretrained된 가중치를 가져왔다
    ```python
    COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"
    with urllib.request.urlopen(COCO_MODEL_URL) as resp, open(coco_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    ```


```python
ROOT_DIR = os.path.abspath("./")
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "./pretrained/mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
MODEL_DIR = os.path.join(ROOT_DIR, "./snapshots/nucleus")
```

## 4.학습을 위한 model 객체 생성 및 model.train()
- Mask_RCNN 패키지는 modellib에 MaskRCNN 객체를 이용하여 Mask RCNN 모델을 생성함. 
- 그렇게 생성한 모델을 이용해서, model.train() model.detect() 를 사용하면 된다.

- 필수 주의 사항 및 생성 인자
    - mode: training인지 inference인지 설정
    - config: training 또는 inference에 따라 다른 config 객체 사용. Config객체를 상속 받아 각 경우에 새로운 객체를 만들고 이를 이용. inference 시에는 Image를 하나씩 입력 받아야 하므로 Batch size를 1로 만들 수 있도록 설정 
    - model_dir: 학습 진행 중에 Weight 모델이 저장되는 장소 지정.   


```python
from mrcnn import model as modellib

train_config = NucleusConfig()
# train_config.display()

model = modellib.MaskRCNN(mode="training", config=train_config,
                                  model_dir=MODEL_DIR)

# Exclude the last layers because they require a matching
# number of classes
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
```

- 학습 데이터 세트와 검증 데이터 세트를 NucleusDataset 객체에 로드하고 train 시작
    - augmentation은 imgaug를 사용. 


```python
from imgaug import augmenters as iaa

img_aug = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])
```

- warning ignore를 안해도 학습은 되지만, imgagu와 충돌하는동안 계속 warning이 발생해서 ignore처리를 하였다. 이런 방법도 있구나 알아두자. 

```python
import warnings 
warnings.filterwarnings('ignore')

print("Train all layers")
model.train(dataset_train, dataset_val,
            learning_rate=train_config.LEARNING_RATE,
            epochs=40, augmentation=img_aug,
            layers='all')
```

    Train all layers
    
    Starting at epoch 0. LR=0.001
    
    Checkpoint Path: /home/sb020518/DLCV/Segmentation/mask_rcnn/./snapshots/nucleus/nucleus20200928T1014/mask_rcnn_nucleus_{epoch:04d}.h5
    Selecting layers to train
    conv1                  (Conv2D)
    bn_conv1               (BatchNorm)
    res2a_branch2a         (Conv2D)
    bn2a_branch2a          (BatchNorm)
    res2a_branch2b         (Conv2D)
    bn2a_branch2b          (BatchNorm)
    res2a_branch2c         (Conv2D)
    res2a_branch1          (Conv2D)
    bn2a_branch2c          (BatchNorm)
    bn2a_branch1           (BatchNorm)
    res2b_branch2a         (Conv2D)
    bn2b_branch2a          (BatchNorm)
    res2b_branch2b         (Conv2D)
    bn2b_branch2b          (BatchNorm)
    res2b_branch2c         (Conv2D)
    bn2b_branch2c          (BatchNorm)
    res2c_branch2a         (Conv2D)
    bn2c_branch2a          (BatchNorm)
    res2c_branch2b         (Conv2D)
    bn2c_branch2b          (BatchNorm)
    res2c_branch2c         (Conv2D)
    bn2c_branch2c          (BatchNorm)
    res3a_branch2a         (Conv2D)
    bn3a_branch2a          (BatchNorm)
    res3a_branch2b         (Conv2D)
    bn3a_branch2b          (BatchNorm)
    res3a_branch2c         (Conv2D)
    res3a_branch1          (Conv2D)
    bn3a_branch2c          (BatchNorm)
    bn3a_branch1           (BatchNorm)
    res3b_branch2a         (Conv2D)
    bn3b_branch2a          (BatchNorm)
    res3b_branch2b         (Conv2D)
    bn3b_branch2b          (BatchNorm)
    res3b_branch2c         (Conv2D)
    bn3b_branch2c          (BatchNorm)
    res3c_branch2a         (Conv2D)
    bn3c_branch2a          (BatchNorm)
    res3c_branch2b         (Conv2D)
    bn3c_branch2b          (BatchNorm)
    res3c_branch2c         (Conv2D)
    bn3c_branch2c          (BatchNorm)
    res3d_branch2a         (Conv2D)
    bn3d_branch2a          (BatchNorm)
    res3d_branch2b         (Conv2D)
    bn3d_branch2b          (BatchNorm)
    res3d_branch2c         (Conv2D)
    bn3d_branch2c          (BatchNorm)
    res4a_branch2a         (Conv2D)
    bn4a_branch2a          (BatchNorm)
    res4a_branch2b         (Conv2D)
    bn4a_branch2b          (BatchNorm)
    res4a_branch2c         (Conv2D)
    res4a_branch1          (Conv2D)
    bn4a_branch2c          (BatchNorm)
    bn4a_branch1           (BatchNorm)
    res4b_branch2a         (Conv2D)
    bn4b_branch2a          (BatchNorm)
    res4b_branch2b         (Conv2D)
    bn4b_branch2b          (BatchNorm)
    res4b_branch2c         (Conv2D)
    bn4b_branch2c          (BatchNorm)
    res4c_branch2a         (Conv2D)
    bn4c_branch2a          (BatchNorm)
    res4c_branch2b         (Conv2D)
    bn4c_branch2b          (BatchNorm)
    res4c_branch2c         (Conv2D)
    bn4c_branch2c          (BatchNorm)
    res4d_branch2a         (Conv2D)
    bn4d_branch2a          (BatchNorm)
    res4d_branch2b         (Conv2D)
    bn4d_branch2b          (BatchNorm)
    res4d_branch2c         (Conv2D)
    bn4d_branch2c          (BatchNorm)
    res4e_branch2a         (Conv2D)
    bn4e_branch2a          (BatchNorm)
    res4e_branch2b         (Conv2D)
    bn4e_branch2b          (BatchNorm)
    res4e_branch2c         (Conv2D)
    bn4e_branch2c          (BatchNorm)
    res4f_branch2a         (Conv2D)
    bn4f_branch2a          (BatchNorm)
    res4f_branch2b         (Conv2D)
    bn4f_branch2b          (BatchNorm)
    res4f_branch2c         (Conv2D)
    bn4f_branch2c          (BatchNorm)
    res5a_branch2a         (Conv2D)
    bn5a_branch2a          (BatchNorm)
    res5a_branch2b         (Conv2D)
    bn5a_branch2b          (BatchNorm)
    res5a_branch2c         (Conv2D)
    res5a_branch1          (Conv2D)
    bn5a_branch2c          (BatchNorm)
    bn5a_branch1           (BatchNorm)
    res5b_branch2a         (Conv2D)
    bn5b_branch2a          (BatchNorm)
    res5b_branch2b         (Conv2D)
    bn5b_branch2b          (BatchNorm)
    res5b_branch2c         (Conv2D)
    bn5b_branch2c          (BatchNorm)
    res5c_branch2a         (Conv2D)
    bn5c_branch2a          (BatchNorm)
    res5c_branch2b         (Conv2D)
    bn5c_branch2b          (BatchNorm)
    res5c_branch2c         (Conv2D)
    bn5c_branch2c          (BatchNorm)
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
    WARNING:tensorflow:From /home/sb020518/anaconda3/envs/tf113/lib/python3.6/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Epoch 1/40
      1/603 [..............................] - ETA: 36:40:48 - loss: 5.2075 - rpn_class_loss: 1.6814 - rpn_bbox_loss: 1.6133 - mrcnn_class_loss: 1.9128 - mrcnn_bbox_loss: 0.0000e

    ---------------------------------------------------------------------------




## 5. 위에서 학습 시킨 모델로 Inference 수행

- 예측용 모델을 로드. mode는 inference로 설정,config는 NucleusInferenceConfig()로 설정,
- 예측용 모델에 위에서 찾은 학습 중 마지막 저장된 weight파일을 로딩함. 
- weight가져온_model.detect() 를 사용하면 쉽게 Inferece 결과를 추출할 수 있다.


```python
class NucleusInferenceConfig(NucleusConfig):
    NAME='nucleus'   # 위의 train config와 같은 name을 가지도록 해야한다.
    # 이미지 한개씩 차례로 inference하므로 batch size를 1로 해야 하며 이를 위해 IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # pad64는 64 scale로 이미지를 맞춰서 resize수행. 
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

```


```python
infer_config = NucleusInferenceConfig()
inference_model = modellib.MaskRCNN(mode="inference", config=infer_config, model_dir=MODEL_DIR)
weights_path = model.find_last()  # model weight path를 직접 정해줘도 되지만, 이와 같은 방법으로, 가장 낮은 loss의 weight path를 찾아준다.
print('학습중 마지막으로 저장된 weight 파일:', weights_path)
inference_model.load_weights(weights_path, by_name=True)

```


```python
- 테스트용 데이터 세트를 NucleusDataset으로 로딩. load_nucleus() 테스트 세트를 지정하는 'stage1_test'를 입력. 
dataset_test = NucleusDataset()
dataset_test.load_nucleus(DATASET_DIR, 'stage1_test') # 위에서 load_nucleus 직접 정의했던거 잊지 말기
dataset_test.prepare()
```


```python
dataset_test.image_ids
"""
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
           34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
           51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64])
"""
```



- 우선 test 셋에 있는 데이터를 model에 통과시켜서, 모든 test data 64개에 대해서, Segmentation결과를 추출하고 visualize 해보자.


```python
for image_id in dataset_test.image_ids:
        # Load image and run detection
        image = dataset_test.load_image(image_id)
        print(len(image))
        # Detect objects
        r = inference_model.detect([image], verbose=0)[0]
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset_test.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
```

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/94464274-3a357b00-01f9-11eb-81f3-f6e3a427957a.png' width='600'/></p>  

```python
for image_id in dataset_test.image_ids:
        # Load image and run detection
        image = dataset_test.load_image(image_id)
        print(len(image))
       
        #plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))
```

## 6. 위 5의 한장의 사진을 inference하는 능력을 가지고 전체사진 inference 수행 하는 함수 만들기


```python
def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = NucleusDataset()
    dataset.load_nucleus(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)
```


```python
detect(model, args.dataset, args.subset)
```
