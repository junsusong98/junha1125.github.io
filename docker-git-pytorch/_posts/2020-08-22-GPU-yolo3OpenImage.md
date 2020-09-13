---
layout: post
title: 【Keras】Keras기반 Yolo3 - Google Open Image 학습시키기 
description: >
    이전 게시물에서 공부했던 Keras 기반 Yolo3를 이용해 Google Open Image 데이터 학습을 시켜보자.
---
 Google Open Image 학습시키기 - Keras기반 Yolo3
 이전의 Pascal VOC, MS coco 데이터 셋에 대해서는 [이전 게시물](https://junha1125.github.io/artificial-intelligence/2020-08-12-detect,segmenta2/) 참조

 

# 1. Google Open Image 고찰 정리
- 600개의 object 카테고리, 
- Object Detection Dataset 170만개 train 4만 validation 12만개 test
- 하지만 데이터셋 용량이 너무 크다보니, COCO로 pretrained된 모델만 전체적으로 공유되는 분위기이다. 
- Version 4 : 2018 Kaggle Challenge Detection Bounding Box
- Version 5 : 2019 Kaggle Challenge Detection Bounding Box + Instace Segmentation
- Version 6 : 2020.02 [https://storage.googleapis.com/openimages/web/index.html](https://storage.googleapis.com/openimages/web/index.html)
- Google Open Image Dataset 특징 : COCO와 비슷하지만, 용량은 훨씬 많다. (아래 이미지 확인)

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/92208999-3565fb80-eec7-11ea-85fd-3bbdbe483bb2.png' alt='drawing' width='600'/></p>

- 아래의 사진처럼, [Download](https://storage.googleapis.com/openimages/web/download_v4.html)를 받아보면 Box의 정보가 **CSV 파일**로 되어있는 것을 확인할 수 있다. (특징 : Xmin Xmax Ymin Ymax가 0~1사이값으로 정규화가 되어있기 때문에 Image w,h를 곱해줘야 정확한 box위치라고 할 수 있다.)

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/92209426-f1272b00-eec7-11ea-89ba-047f7f1fddd7.png' alt='drawing' width='600'/></p>

- 아래의 사진처럼, Metadate의 Class Names를 확인해보면 다음과 같이 ClassID와 Class 이름이 mapping되어 있는 것을 확인할 수 있다. (나중에 사용할 예정)

- Class 구조([Hierarchy_bounding box label](https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy_visualizer/circle.html)). (Download 사이트의 맨 아래로 가면 볼 수 있다.)

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/92210129-1ff1d100-eec9-11ea-8129-3cbbbca34bbd.png' alt='drawing' width='230'/></p>


# 2. OIDv4 Toolkit 
- [OIDv4 Toolkit Git 사이트](https://github.com/EscVM/OIDv4_ToolKit) 
- 방대한 Open Image를 전체 다운로드 받는 것은 힘드므로, 원하는 양의 이미지, 원하는 클래스의 이미지만 다운로드 할 수 있게 해주는 툴이다. 
-  Xmin Xmax Ymin Ymax가 0~1사이값을 좌표로 자동 변환을 해준다. 
- 설치  
    ```sh
    $ cd ~/DLCV/data
    $ git clone https://github.com/EscVM/OIDv4_ToolKit.git
    $ pip3 install -r requirements.txt
    ```
- 다운로드   - 아래의 1,2,3,4 순서과정 주의하기
    ```sh
    $ python3 main.py downloader --classes Apple Orange --type_csv validation

    아래와 같은 예시로 다운로드 된다. 
     main_folder
    │   main.py
    │
    └───OID
        │   file011.txt
        │   file012.txt
        │
        └───csv_folder
        |    │   class-descriptions-boxable.csv         1. open image Metadata에 있는 class 이름
        |    │   validation-annotations-bbox.csv        2. open image의 annotation csv파일 그대로 다운로드
        |
        └───Dataset
            |
            └─── test
            |
            └─── train
            |
            └─── validation                         3. 위의 anotation csv 파일을 참조해서 apple과 orange에 관련된 이미지만 아래와 같이 다운로드 
                |
                └───Apple
                |     |
                |     |0fdea8a716155a8e.jpg
                |     |2fe4f21e409f0a56.jpg
                |     |...
                |     └───Labels
                |            |
                |            |0fdea8a716155a8e.txt  4. 사과에 대한 내용만 annotation으로 csv 파일형식으로 txt파일로 만들어 놓는다. 
                |            |2fe4f21e409f0a56.txt
                |            |...
                |
                └───Orange
                    |
                    |0b6f22bf3b586889.jpg
                    |0baea327f06f8afb.jpg
                    |...
                    └───Labels
                            |
                            |0b6f22bf3b586889.txt
                            |0baea327f06f8afb.txt
                            |...
    ```  
- 위의 명령어에서 validation은 Google Open Image에서 validation에 있는 데이터 중에 사과와 오렌지에 관한 것만 다운로드를 해준다. 
- 직접 쳐서 다운로드 하고 어떻게 구성되어 있는지 직접 확인하기. 

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/92255868-e2636700-ef0d-11ea-8cfb-46beab97f3fb.png' alt='drawing' width='600'/></p>

- Keras-Yolo3에서 데이터를 활용하기 위한 변환 과정 (이미지 있는 코드 적극 활용 위해)
    1. OIDv4 Toolkit으로 데이터 다운로드
    2. VOC XML 타입으로 변경
    3. Keras yolo용 CSV로 변환
    

# 3. 데이터 다운로드 및 전처리 해보기(Keras-yolo3 용)

- 아래 전체 과정 요약
    1. 내가 가진 데이터 전처리(패키지가 원하는 형식으로)
    2. Data Genereator에 넣기(torch같은 경우 TensorDataset, DataLoader를 이용해서)
    3. create_model 하기. (torch.nn 모듈을 사용해서 정의한 신경망 모델 클래스로, 객체 생성하기)
    4. Check point, log 생성을 위한 작업하기.(Keras 같은 경우 keras.callbacks 모듈의 함수 이용)
    5. model을 위한 optimizer, Loss function(criterion) 설정하기. 
    6. model.fit_generator를 이용해서 학습 및 가중치 갱신 시키기. (from keras.models import Model에 있는 맴버함수이다.)
    7. .h5 형식의 Inference를 위한 가중치 파일을 얻을 수 있다.
    8. .h5 형식 파일을 이용해서, yolo.py함수의 YOLO 클래스를 이용해 객체 생성
    9. YOLO.detect_image를 이용해서 객체 탐지 수행

1. 실습을 위한 데이터 다운로드
    ```sh
    $ python3 main.py downloader --classes Football Football_helmet Fish Shark Shellfish  --type_csv train --multiclasses 1 --limit 300
    # multiclasses : 다양한 class의 이미지를 위에처럼 다른 폴더에 넣는게 아니라 하나의 폴더에 묶어서 다운로드
    # limit : 카테고리당, 최대 이미지 수
    # Football_helmet라고 쳐도 되고, 'Football helmet'라고 해도 된다.
    ```

2. Label 폴더의 annotation.txt 파일을 모두 VOC XML 파일로 바꾼다. 
    - Label 폴더의 annotation.txt 파일들이 각각 이미지 하나에 txt파일 이미지 하나로 맴핑된다. 
    - 이것은 VOC XML 파일 형식과 매우 유사한 것이다. (이미지 하나당, XML 파일 하나)
    - 따라서 annotation.txt 파일을 XML 파일로 바꾼다.
    - Git에 돌아니는, oid_to_pascal_voc_xml.py 파일을 이용할 것이다. 해당 파일을 보면 알겠듯이, OIDv4_ToolKit repository가 있는 그 위치에 파일이 존재해야한다.

    ```sh
    $ cd ~/DLCV/data/util
    $ cp oid_to_pascal_voc_xml.py ~/DLCV/data/OIDv4_ToolKit
    $ cd ~/DLCV/data/OIDv4_ToolKit
    $ python oid_to_pascal_voc_xml.py 
    ```
    
    - 아래처럼 xml파일이 /train/ballnfish 파일 바로 아래에 jpg파일과 같은 위치에 생성된 것을 알 수 있다. 

    <p align="center"><img src='https://user-images.githubusercontent.com/46951365/92258246-7edb3880-ef11-11ea-8def-6a062908a81a.png' alt='drawing' width='600'/></p>

3. [이전 Post 내용(라쿤데이터셋 전처리)](https://junha1125.github.io/docker-git-pytorch/2020-08-20-GPU-yolo3Raccoon/#3-1-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC-%ED%95%98%EA%B8%B0)을 활용해서 위의 XML파일을 CSV로 전환
    - 새로운 위치에 이미지와 xml파일 옮기기 

    ```sh
    $ cd ~/DLCV/data
    $ mkdir ballnfish
    $ cd ./ballnfish
    $ mkdir annotation
    $ mkdir images
    $ cd ~/DLCV/data/OIDv4_ToolKit/OID/Dataset/train/ballnfish 
    $ cp *.jpg  ~/DLCV/data/ballnfish/images
    $ cp *.xml  ~/DLCV/data/ballnfish/annotation
    ```

    - images에 들어가면 이미지가 1499개 있다. (하나의 카테코리당 300개 이미지)
    - [이전에 했던 라쿤데이터셋 전처리 함수](https://junha1125.github.io/docker-git-pytorch/2020-08-20-GPU-yolo3Raccoon/#3-1-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC-%ED%95%98%EA%B8%B0) 이용하기 xml_to_csv 함수 찾아 이용.
    - 여기서부터 참조 : /DLCV/Detection/yolo/KerasYolo_OpenImage_학습및_Detection.ipynb

- Open Image Dataset의 Object Detection 학습 및 Inference
    * Open Image Dataset에서 Football 관련 Object, Fish관련 Object를 추출 후 학습 데이터 세트 생성. 
    * 이를 이용하여 Object Detection 수행. 


```python
# annotation과 image 디렉토리 설정. annotation디렉토리에 있는 파일 확인. 
import os
from pathlib import Path

HOME_DIR = str(Path.home())

ANNO_DIR = os.path.join(HOME_DIR, 'DLCV/data/ballnfish/annotations')
IMAGE_DIR = os.path.join(HOME_DIR, 'DLCV/data/ballnfish/images')
print(ANNO_DIR)

files = os.listdir(ANNO_DIR)
print('파일 개수는:',len(files))
print(files)
```


```python
!cat /home/sb020518/DLCV/data/ballnfish/annotations/e5e4aba50208e92f.xml
```

    <annotation>
      <folder>vallnfish</folder>
      <filename>e5e4aba50208e92f.jpg</filename>
      <path>/home/sb020518/DLCV/data/OIDv4_ToolKit/OID/Dataset/train/vallnfish/e5e4aba50208e92f.jpg</path>
      ...
      <object>
        <name>Football_helmet</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
          <xmin>685</xmin>
          <ymin>86</ymin>
          <xmax>778</xmax>
          <ymax>197</ymax>
        </bndbox>
      </object>
    </annotation>



```python
import glob
import xml.etree.ElementTree as ET

classes_map = {'Football':0, 'Football_helmet':1, 'Fish':2, 'Shark':3, 'Shellfish':4 }

def xml_to_csv(path, output_filename):
    xml_list = []
    # xml 확장자를 가진 모든 파일의 절대 경로로 xml_file할당. 
    with open(output_filename, "w") as train_csv_file:
        for xml_file in glob.glob(path + '/*.xml'):
            # xml 파일을 parsing하여 XML Element형태의 Element Tree를 생성하여 object 정보를 추출. 
            tree = ET.parse(xml_file)
            root = tree.getroot()
            # 파일내에 있는 모든 object Element를 찾음. 
            full_image_name = os.path.join(IMAGE_DIR, root.find('filename').text)
            value_str_list = ' '
            for obj in root.findall('object'):
                xmlbox = obj.find('bndbox')
                class_name = obj.find('name').text
                x1 = int(xmlbox.find('xmin').text)
                y1 = int(xmlbox.find('ymin').text)
                x2 = int(xmlbox.find('xmax').text)
                y2 = int(xmlbox.find('ymax').text)
                # 
                class_id = classes_map[class_name]
                value_str = ('{0},{1},{2},{3},{4}').format(x1, y1, x2, y2, class_id)
                # object별 정보를 tuple형태로 object_list에 저장. 
                value_str_list = value_str_list+value_str+' '
        
            train_csv_file.write(full_image_name+' '+ value_str_list+'\n')
        # xml file 찾는 for loop 종료 
```


```python
xml_to_csv(ANNO_DIR, os.path.join(ANNO_DIR,'ballnfish_anno.csv'))
print(os.path.join(ANNO_DIR,'ballnfish_anno.csv'))
```

```python
!cat /home/sb020518/DLCV/data/ballnfish/annotations/ballnfish_anno.csv
```
    /home/sb020518/DLCV/data/ballnfish/images/6d3552b1e28f05c0.jpg  339,304,631,472,2 
    /home/sb020518/DLCV/data/ballnfish/images/379a8df86070df59.jpg  361,312,498,445,1 76,39,108,77,1 120,42,171,92,1 
    /home/sb020518/DLCV/data/ballnfish/images/ba993dcfb21071c7.jpg  177,345,753,678,2 
    /home/sb020518/DLCV/data/ballnfish/images/1a51f73f96d2e88b.jpg  50,294,778,598,3 
    /home/sb020518/DLCV/data/ballnfish/images/823873ff2c000a52.jpg  126,127,912,484,2 
    /home/sb020518/DLCV/data/ballnfish/images/022ab76a12ee9808.jpg  0,0,1023,767,4 445,408,771,692,4 
    /home/sb020518/DLCV/data/ballnfish/images/222e79f02c455fc5.jpg  64,5,942,613,3 
    /home/sb020518/DLCV/data/ballnfish/images/0eade5d90ecc799e.jpg  111,35,139,53,1 135,85,175,116,1 170,25,200,42,1 225,42,255,71,1 264,100,309,149,1 280,46,304,67,1 479,37,510,71,1 552,55,588,81,1 604,81,637,113,1 709,69,746,109,1 
    /home/sb020518/DLCV/data/ballnfish/images/de93f11430812481.jpg  445,256,566,346,2 264,443,307,509,2 553,320,658,376,2 796,251,865,289,2 
    /home/sb020518/DLCV/data/ballnfish/images/94798692e8a62998.jpg  131,162,997,741,2 
    /home/sb020518/DLCV/data/ballnfish/images/aff435118770ffce.jpg  355,208,753,530,3 

# 4. 모델 구성하고 학습시키기

- 라쿤 데이터 학습과는 조금 많이 다르게 학습시킨다.
- 좀 더 하나하나를 함수화 해서 체계적으로 코드를 구성해 나갈 계획이다. 


```python
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

plt.imshow(cv2.cvtColor(cv2.imread('/home/sb020518/DLCV/data/ballnfish/images/a9b21c816203e357.jpg'), cv2.COLOR_BGR2RGB))
```




```python
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
```


```python
import sys, os

LOCAL_PACKAGE_DIR = os.path.abspath("./keras-yolo3")
sys.path.append(LOCAL_PACKAGE_DIR)

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
```


```python
from train import get_classes, get_anchors
from train import create_model, data_generator, data_generator_wrapper

BASE_DIR = os.path.join(HOME_DIR, 'DLCV/Detection/yolo/keras-yolo3')

## 학습을 위한 기반 환경 설정. annotation 파일 위치, epochs시 저장된 모델 파일, Object클래스 파일, anchor 파일.
annotation_path = os.path.join(ANNO_DIR, 'ballnfish_anno.csv')
log_dir = os.path.join(BASE_DIR, 'snapshots/ballnfish/')
classes_path = os.path.join(BASE_DIR, 'model_data/ballnfish_classes.txt')
anchors_path = os.path.join(BASE_DIR,'model_data/yolo_anchors.txt')

class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
print(class_names, num_classes)
print(anchors)
```

    ['Football', 'Football_helmet', 'Fish', 'Shark', 'Shellfish'] 5
    [[ 10.  13.]
     [ 16.  30.]
     [ 33.  23.]
     [ 30.  61.]
     [ 62.  45.]
     [ 59. 119.]
     [116.  90.]
     [156. 198.]
     [373. 326.]]


- yolo 모델 학습을 위한 전반적인 파라미터를 config 클래스로 설정하고 필요시 이를 수정하여 학습. 


```python
# csv annotation 파일을 읽어서 lines 리스트로 만듬. 
with open(annotation_path) as f:
    lines = f.readlines()

# 추후에 사용할 config 정보를 json, xml 등으로 적을 수 있겠지만, 여기서는 class변수 하나에 config에 대한 정보를 모아 놓았다. 
# 해당 cell의 맨 아래와 같이 맴버변수를 사용해서 환경변수 데이터를 불러올 예정이다.
class config:
    #tiny yolo로 모델로 초기 weight 학습 원할 시 아래를 tiny-yolo.h5로 수정. 
    initial_weights_path=os.path.join(BASE_DIR, 'model_data/yolo.h5' )
    # input_shape는 고정. 
    input_shape=(416, 416) 
    # epochs는 freeze, unfreeze 2 step에 따라 설정. 
    first_epochs=50
    first_initial_epochs=0
    second_epochs=100
    second_initial_epochs=50
    # 학습시 batch size, train,valid건수, epoch steps 횟수  
    batch_size = 2
    val_split = 0.1   
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    train_epoch_steps = num_train//batch_size 
    val_epoch_steps =  num_val//batch_size
    
    anchors = get_anchors(anchors_path)
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    # epoch시 저장된 weight 파일 디렉토리 
    log_dir = os.path.join(BASE_DIR, 'snapshots/ballnfish/')  
    
print('Class name:', config.class_names,'\nNum classes:', config.num_classes)
```

    Class name: ['Football', 'Football_helmet', 'Fish', 'Shark', 'Shellfish'] 
    Num classes: 5


- create_generator 함수 만들기
    - csv 파일을 입력 받아서 train 데이터와 valid 데이터 처리를 위한 data_generator_wrapper객체를 각각 생성.
    - train용, valid 용 data_generator_wrapper는 Yolo 모델의 fit_generator()학습시 인자로 입력됨. 


```python
def create_generator(lines):
    
    train_data_generator = data_generator_wrapper(lines[:config.num_train], config.batch_size, 
                                                  config.input_shape, config.anchors, config.num_classes)
    
    valid_data_generator = data_generator_wrapper(lines[config.num_train:], config.batch_size, 
                                                  config.input_shape, config.anchors, config.num_classes)
    
    return train_data_generator, valid_data_generator
```

- create_yolo_model 함수 만들기
    - YOLO 모델 또는 tiny yolo 모델 반환. 초기 weight값은 pretrained된 yolo weight값으로 할당. 


```python
# anchor 개수에 따라 tiny yolo 모델 또는 yolo 모델 반환. 
# tiny yolo 를 위한 config파일은 나중에 만들어 사용할 에정이다.
def create_yolo_model():
    is_tiny_version = len(config.anchors)==6 
    if is_tiny_version:
        model = create_tiny_model(config.input_shape, config.anchors, config.num_classes, 
            freeze_body=2, weights_path=config.initial_weights_path)
    else:
        model = create_model(config.input_shape, config.anchors, config.num_classes, 
            freeze_body=2, weights_path=config.initial_weights_path)
        
    return model 
```

- callback 객체들을 생성. 
    - Keras에서 많이 사용하는 checkpoint 저장 방식인듯 하다. 우선 이것도 모르지만 넘어가자.


```python
# Tensorboard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping callback 반환
def create_callbacks():
    logging = TensorBoard(log_dir=config.log_dir)
    checkpoint = ModelCheckpoint(config.log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)  # epoch 3마다 weight 저장
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    # early_stopping에 의해 loss가 항상 비슷하면 학습을 자동 중단한다. patience를 늘리면 더 학습이 진행된다.
    
    #개별 callback들을 한꺼번에 list로 묶어서 반환
    return [logging, checkpoint, reduce_lr, early_stopping]
```

- 위의 작업으로 모델 구성을 마쳤다. 학습 수행을 시작한다 {:.lead}
    - GPU : \$ watch -d -n 1 nvidia-smi
    - CPU : \$ vmstat 3   
        us : 사용자 사용량 wa : wait cpu가 놀고 있는 량
        
<p align="center"><img src='https://user-images.githubusercontent.com/46951365/92294435-33537980-ef67-11ea-8237-d3d391832023.png' alt='drawing' width='500'/></p>


```python
   
# create_generator(), create_model(), create_callbacks() 수행. 
train_data_generator, valid_data_generator = create_generator(lines)
ballnfish_model = create_yolo_model()
callbacks_list = create_callbacks()

# 최초 모델은 주요 layer가 freeze되어 있음. 안정적인 loss를 확보하기 위해 주요 layer를 freeze한 상태로 먼저 학습. 
print('First train 시작' )
ballnfish_model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

#1단계 학습 완료 모델 저장. 
# 아래의 과정을 통해서 foward -> backpropagation -> weight 갱신 -> weight 저장
ballnfish_model.fit_generator(
                train_data_generator, 
                steps_per_epoch=config.train_epoch_steps,
                validation_data=valid_data_generator, 
                validation_steps=config.val_epoch_steps,
                epochs=config.first_epochs, 
                initial_epoch=config.first_initial_epochs, 
                callbacks=callbacks_list)
ballnfish_model.save_weights(log_dir + 'trained_weights_stage_1.h5')

# 2단계 학습 시작. 
# create_model() 로 반환된 yolo모델에서 trainable=False로 되어 있는 layer들 없이, 모두 True로 만들고 다시 학습 (뭔지 정확히 모르겠음)
# 모든 layer를 trainable=True로 설정하고 학습 수행. 
for i in range(len(ballnfish_model.layers)):
    ballnfish_model.layers[i].trainable = True
    
print('Second train 시작' )
ballnfish_model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) 
ballnfish_model.fit_generator(train_data_generator, steps_per_epoch=config.train_epoch_steps,
    validation_data=valid_data_generator, validation_steps=config.val_epoch_steps,
    epochs=config.second_epochs, initial_epoch=config.second_initial_epochs,
    callbacks=callbacks_list)

# 최종 학습 완료 모델 저장. 
ballnfish_model.save_weights(log_dir + 'trained_weights_final.h5')
```

- 최종 학습된 모델을 로딩하여 Object Detection 수행. 
    - 이제 위에서 만든 구성 및 모듈 다 필요없다. 
    - 가장 마지막에 만들어진 Weight 파일인 /keras-yolo3/snapshots/ballnfish/trained_weights_final.h5 를 가지고 학습을 시작하자


```python
from yolo import YOLO
#keras-yolo에서 image처리를 주요 PIL로 수행. 
from PIL import Image

LOCAL_PACKAGE_DIR = os.path.abspath("./keras-yolo3")
sys.path.append(LOCAL_PACKAGE_DIR)0

ballnfish_yolo = YOLO(model_path='/home/younggi.kim999/DLCV/Detection/yolo/keras-yolo3/snapshots/ballnfish/trained_weights_final.h5',
            anchors_path='~/DLCV/Detection/yolo/keras-yolo3/model_data/yolo_anchors.txt',
            classes_path='~/DLCV/Detection/yolo/keras-yolo3/model_data/ballnfish_classes.txt')
```

- 이미지 Object Detection
    - 위에서 정의한 모델을 가지고 **ballnfish_yolo.detect_image(img)** 를 사용해서 detection 수행해 나가자!


```python
football_list = ['f1b492a9bce3ac9a.jpg', '1e6ff631bb0c198b.jpg', '97ac013310bda756.jpg',
                'e5b1646c395aecfd.jpg', '53ef241dad498f6c.jpg', '02ccbf5ddaaecedb.jpg' ]
for image_name in football_list:
    img = Image.open(os.path.join(IMAGE_DIR, image_name))
    detected_img = ballnfish_yolo.detect_image(img)
    plt.figure(figsize=(8, 8))
    plt.imshow(detected_img)
    
```


```python
helmet_list = ['1fed5c930211c6e0.jpg', '011a59a160d7a091.jpg', 'd39b46aa4bc0c165.jpg', '7e9eb7eba80e34e7.jpg', '9c27811a78b74a48.jpg']
for image_name in helmet_list:
    img = Image.open(os.path.join(IMAGE_DIR, image_name))
    detected_img = ballnfish_yolo.detect_image(img)
    plt.figure(figsize=(8, 8))
    plt.imshow(detected_img)
```


```python
fish_list = ['25e42c55bfcbaa88.jpg', 'a571e4cdcfbcb79e.jpg', '872c435491f2b4d3.jpg', 
             'bebac23c45451d93.jpg', 'eba7caf07a26829b.jpg', 'dc607a2989bdc9dc.jpg' ]
for image_name in fish_list:
    img = Image.open(os.path.join(IMAGE_DIR, image_name))
    detected_img = ballnfish_yolo.detect_image(img)
    plt.figure(figsize=(8, 8))
    plt.imshow(detected_img)
```


```python
shark_list = ['d92290f6c04dd83b.jpg', '3a37a09ec201cdeb.jpg', '32717894b5ce0052.jpg', 'a848df5dbed78a0f.jpg', '3283eafe11a847c3.jpg']
for image_name in shark_list:
    img = Image.open(os.path.join(IMAGE_DIR, image_name))
    detected_img = ballnfish_yolo.detect_image(img)
    plt.figure(figsize=(8, 8))
    plt.imshow(detected_img)
```


```python
shell_list=['5cc89bc28084e8e8.jpg',  '055e756883766e1f.jpg', '089354fc39f5d82d.jpg', '80eddfdcb3384458.jpg']
for image_name in shell_list:
    img = Image.open(os.path.join(IMAGE_DIR, image_name))
    detected_img = ballnfish_yolo.detect_image(img)
    plt.figure(figsize=(8, 8))
    plt.imshow(detected_img)
```

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/92295417-7feb8480-ef68-11ea-9e8d-65c082b253ac.png' alt='drawing' width='700'/></p>

# 5. 영상 Object Detection 

- 어렵지 않게 **cv2.VideoCapture** 함수와 **ballnfish_yolo.detect_image(img)** 를 사용해서 Detection 수행하면 끝!


```python
import cv2
import time

def detect_video_yolo(model, input_path, output_path=""):
    
    start = time.time()
    cap = cv2.VideoCapture(input_path)
    
    #codec = cv2.VideoWriter_fourcc(*'DIVX')
    codec = cv2.VideoWriter_fourcc(*'XVID')
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    vid_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vid_writer = cv2.VideoWriter(output_path, codec, vid_fps, vid_size)
    
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('총 Frame 갯수:', frame_cnt, '원본 영상 FPS:',vid_fps, '원본 Frame 크기:', vid_size)
    index = 0
    while True:
        hasFrame, image_frame = cap.read()
        if not hasFrame:
            print('프레임이 없거나 종료 되었습니다.')
            break
        start = time.time()
        # PIL Package를 내부에서 사용하므로 cv2에서 읽은 image_frame array를 다시 PIL의 Image형태로 변환해야 함.  
        image = Image.fromarray(image_frame)
        # 아래는 인자로 입력된 yolo객체의 detect_image()로 변환한다.
        detected_image = model.detect_image(image)
        # cv2의 video writer로 출력하기 위해 다시 PIL의 Image형태를 array형태로 변환 
        result = np.asarray(detected_image)
        index +=1
        print('#### frame:{0} 이미지 처리시간:{1}'.format(index, round(time.time()-start,3)))
        
        vid_writer.write(result)
    
    vid_writer.release()
    cap.release()
    print('### Video Detect 총 수행시간:', round(time.time()-start, 5))
```


```python
detect_video_yolo(ballnfish_yolo, '../../data/video/NFL01.mp4', '../../data/output/NFL_yolo_01.avi')
!gsutil cp ../../data/output/NFL_yolo_01.avi gs://my_bucket_dlcv/data/output/NFL_yolo_01.avi
```


```python
detect_video_yolo(ballnfish_yolo, '../../data/video/FishnShark01.mp4', '../../data/output/FishnShark_yolo_01.avi')
!gsutil cp ../../data/output/FishnShark_yolo_01.avi gs://my_bucket_dlcv/data/output/FishnShark_yolo_01.avi
```

- 분석결과 : 영상의 FPS가 작다. 그래서 빠르게 움직이는 물체들을 잡는 능력이 부족하다. 
- 영상의 FPS가 커지면, Model의 Inference 속도도 굉장히 빨라져야한다....
- 쉽지 않은 문제이다. 

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/92295812-0bfdac00-ef69-11ea-81c8-d35ce4d12b74.png' alt='drawing' width='600'/></p>

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/92295858-980fd380-ef69-11ea-809d-a6d56a38a2e9.png' alt='drawing' width='600'/></p>


