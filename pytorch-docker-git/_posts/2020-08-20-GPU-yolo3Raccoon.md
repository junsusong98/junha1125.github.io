---
layout: post
title: 【Keras】Keras기반 Yolo3 -라쿤 데이터 학습시키기 + GPU 학습 유의사항
description: >
    GPU Object Detection시 유의 사항을 간단히 알아보고, 이전 게시물에서 공부했던 Keras 기반 Yolo3를 이용해 데이터 학습을 시켜보자.
---
GPU 학습 유의사항/ 라쿤 데이터 학습시키기 - Keras기반 Yolo3 

# 1. GPU 학습 시킬 떄 주의 사항
1. 대량의 이미지 학습시 메모리 문제
    - 알다시피 개별 이미지가 Tensor, narray로 변환해서 NeuralNet에 집어넣을떄 이미지를 Batch로 묶어 처리한다. 
    - 이 Batch가 그래픽 카드 메모리 사용량의 많은 부분을 좌우한다.  
    (1000장 - 10장 배치 * 100번 = 1 에폭)
    - GPU 사용 % 자주 확인하기. Image를 loading하는 것은 CPU에서 한다. Batch가 크면 GPU는 바쁘고 CPU는 논다. Batch가 작으면 GPU는 놀고 CPU는 바쁘다. 근데 Object Detection은 GPU에서 할게 너무 많아서 Batch를 작게하는게 대부분이다. (Batch = 1~16) 
    
2. Genrator와 Batch
    - 특히 fit_generator를 확인해보면, Python gererator를 사용한다. gerator, next, yield에 대해서 알아보자. [코딩도장 generator 설명](https://dojang.io/mod/page/view.php?id=2412)
    - 배치 사이즈를 크게 늘린다고 학습이나 추론이 빨라지지 않는다. 100장의 이미지를 한방에 끌어오는것도 힘들지만, 그 이미지를 네트워크에 넣는다 하더라도 어딘가에서 병목이 걸릴건 당연하다. 따라서 배치 사이즈를 어떻게 설정하지는 CPU core, GPU 성능 등을 파악해서 균형있게 맞춰야 한다.(Heuristic 하게 Turning)
    - 한쪽은 열심히 일하고, 한쪽은 기다리는 현상이 일어나지 않게끔. 이렇게 양쪽이 일을 쉬지 않고 균형있게 배치하는 것이 필요하다. 
    - **이러한 Batch를 Keras에서는 (fit_generator 함수가 사용하는) DataGenerator, flow_from_drectory 가 해준다. 알다시피 Torch는 TensorDataset, DataLoader가 해준다.**
    - 이런 generator를 사용하기 위해서 Keras와 같은 경우에 다음과 같은 코드를 사용한다. 
    
        ```python
            from keras.preprocessing.image import ImageDataGenerator

            train_datagen = ImageDataGerator(rescale=1/255) # 이미지 정규화
            train_gernator = train_datagen.flow_from_director('학습용 이미지 Directory', targer_size = (240,240), batch_size-10, class_mode='categoricl') 
            valid_datagen = ImageDataGerator(rescale=1/255) 
            valid_gernator = valid_datagen.flow_from_director('검증용 이미지 Directory', targer_size = (240,240), batch_size-10, class_mode='categoricl')

            # 모델을 구성하고
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(240,240,3)))
            model.add ...
            ...
            ...
            model.add(Dense(3, activation='softmax'))
            model.compile(loss='categorial_crossentropy', optimizer='adam', metrixs=['accuracy'])

            # 학습을 시킨다.
            model.fit_generator(train_generator, steps_per_epoch=100, epochs=10, validation_data=valid_generator, validation_steps=2)
        ```
    - 이렇게 fit_generator을 돌리면 data pipeline이 만들어 진다. train_generator가 디렉토리에 가서 배치만큼의 파일을 읽어와서 yeild를 하여 모델에 데이터를 넣는다. 


    <p align="center"><img src='https://user-images.githubusercontent.com/46951365/92118150-0bfa9080-ee31-11ea-8398-6de32d892007.png' alt='drawing' width='800'/></p>


# 2. qqwweee/Keras-Yolo3로 Training 하기 위한 분석
주의사항{:.lead}
- 아래의 일렬의 과정들은 패키지의 있는 함수들을 전적으로 사용합니다. 
- 따라서 그냥 보면 흐름을 이해하지 못할 수도 있습니다. 
- 꼭 패키지의 코드들과 같이 보는 것을 추천합니다.
- 만약 케라스를 잘 모른다면, 그냥 다른 Data를 학습시킬때는 이런 일렬의 과정들이 있구나  
  정도의 감을 잡는 것도 아주 좋은 공부 일 수 있습니다.
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

- 지금까지는 1. Pretrained된 Weight값을 가져와서 Inference를 수행하거나(Tensorflow(SSD, Yolo), Keras-yolo3) 2. OpenCV DNN 모듈을 사용하거나 해왔다.
- 참고 : 1개의 객체만 검출한다면 100장이어도 충분하다. 근데 Class가 여러개면 수 많은 이미지 데이터가 필요하다. 
- \<weight 가중치 파일 형식 정리>
    - **keras weight(케라스 가중치) 파일은 파일 형식이 .h5**파일이다. 
    - Tensorflow에서 Inference를 위한 파일 형식은 .pb 파일이다. .ckpt파일에는 학습에 대한check point에 대한 정보를 담고 있다. 
    - PyTorch에서는 .pt 파일이 torch.save(model.state_dict()를 이용해서 저장한 weight파일이다. 
- qqwweee/keras-yolo3에 적힌 Training 방법 정리
    1. voc(xml 파일사용)을 annotation파일로 바꾸는 convert_annotation.py 파일을 살펴보면 최종적으로 '%s_%s.txt'%(year, image_set) 파일을 만드는 것을 확인할 수 있다. 
    2. Readme의 내용을 따르면,  
    각 한줄 : image_file_path box1 box2 ... boxN  
    boxK의 format : x_min,y_min,x_max,y_max,class_id (no space) 이다.   
        ```sh
        path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
        path/to/img2.jpg 120,300,250,600,2
        ...
        ```  
    3. 일단 pretrained된 weight를 가져와서 학습시키는 것이 좋으니, conver.py를 실행해서 darknet weight를 keras weight로 바꿔준다.   
    4. train.py를 사용해서 학습을 한다. 내부 코드 적극 활용할 것.
    5. train.py에 들어가야하는 Option을 설정하는 방법은 원래의 default option이 무엇인지 확인하면 된다. 

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/92137843-a31f1280-ee48-11ea-9094-84e267048ef3.png' alt='drawing' width='300'/></p>

- train.py을 적극적으로 이용할 계획이다. 아래와 같은 방법으로. 함수는 아래와 같이 import를 해서 사용할 것이고, main함수의 내용은 소스 코드 그대로 복사해 가져와서 사용할 예정이다.   
    ```python
        from train import get_classes, get_anchors
        from train import create_model, data_generator 
    ```


# 3. 라쿤 데이터셋으로 학습시키고 Inference하기

- 참조 : /DLCV/Detection/yolo/KerasYolo_Raccoon_학습및_Detection.ipynb
- [좋은 데이터 세트 모음](https://github.com/experiencor/keras-yolo3) : [라쿤데이터셋](https://github.com/experiencor/raccoon_dataset)

### 3-1 데이터 전처리 하기  
- Raccoon 데이터 세트 download 후
    - 이미지와 annoation 디렉토리를 제외하고 모두 삭제합니다. 
    - images에 이미지가 있고 annoation에는 각이미지에 해당하는 xml파일 203개가 존재한다. 
- **아래의 내용들은 전처리를 하기에 아주 좋은 함수들과 모듈들이 잘 사용되고 있으므로 잘 알아두고 나중에도 꼭 잘 활용하자**


```python
# annotation과 image 디렉토리 설정. annotation디렉토리에 있는 파일 확인. 
import os
from pathlib import Path

HOME_DIR = str(Path.home())

ANNO_DIR = os.path.join(HOME_DIR, 'DLCV/data/raccoon/annotations')
IMAGE_DIR = os.path.join(HOME_DIR, 'DLCV/data/raccoon/images')
print(ANNO_DIR)

files = os.listdir(ANNO_DIR)
print('파일 개수는:',len(files))  # 200개
print(files)  # Dicrectory에 있는 파일 이름 전부가 list 변수에 저장된다.
```

- 아래에서 사용할 모듈 참고 문헌 (해당 사이트만 공부하면 아래의 내용들 충분히 커버 가능)
    1. glob - [https://wikidocs.net/83](https://wikidocs.net/83)
    1. xml.etree.ElementTree as EF - [http://egloos.zum.com/sweeper/v/3045370](http://egloos.zum.com/sweeper/v/3045370)


```python
import glob
import xml.etree.ElementTree as ET

def xml_to_csv(path, output_filename):
    """
    path : annotation Detectory
    filename : ouptut file name
    """
    xml_list = []
    # xml 확장자를 가진 모든 파일의 절대 경로로 xml_file할당. 
    with open(output_filename, "w") as train_csv_file:
        for xml_file in glob.glob(path + '/*.xml'):
            # path에 있는 xml파일중 하나 하나를 가져온다. 
            tree = ET.parse(xml_file) 
            root = tree.getroot()
            # 파일내에 있는 모든 object Element를 찾음. 
            full_image_name = os.path.join(IMAGE_DIR, root.find('filename').text)
            value_str_list = ' '
            # find all <object>인것 다 찾는다
            for obj in root.findall('object'): 
                xmlbox = obj.find('bndbox')
                x1 = int(xmlbox.find('xmin').text)
                y1 = int(xmlbox.find('ymin').text)
                x2 = int(xmlbox.find('xmax').text)
                y2 = int(xmlbox.find('ymax').text)
                # 단 하나의 class_id raccoon
                class_id = 0
                value_str = ('{0},{1},{2},{3},{4}').format(x1, y1, x2, y2, class_id)
                value_str_list = value_str_list+value_str+' ' 
                # box1 box2 ......
                # object별 정보를 tuple형태로 object_list에 저장. 
            train_csv_file.write(full_image_name+' '+ value_str_list+'\n') # image_file_path box1 box2 ... boxN \n ... image_file_path
        # xml file 찾는 for loop 종료 
```


```python
xml_to_csv(ANNO_DIR, os.path.join(ANNO_DIR,'raccoon_anno.csv'))
print(os.path.join(ANNO_DIR,'raccoon_anno.csv'))
```

    /home/sb020518/DLCV/data/raccoon/annotations/raccoon_anno.csv


### 3-2 전처리한 데이터로 학습시키기 - pretrain 모델가져오기

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/92196552-7f8cb400-eeaa-11ea-9074-601a4a614b6b.png' alt='drawing' width='400'/></p>

- 위의 파일 내부의 함수들을 import할 예정이다. import방법은 아래와 같다. 참조하자
- 사진의 model.py, utils.py, keras-yolo3의 train.py파일을 아래와 같이 import할 것이다. 


```python
import sys

LOCAL_PACKAGE_DIR = os.path.abspath("./keras-yolo3")
sys.path.append(LOCAL_PACKAGE_DIR)

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data 
# get_random_data - Image Augmentation 하기 위한 모듈 - Augmentation을 할 때 Bounding Box 처리도 해준다.
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
from train import get_classes, get_anchors
from train import create_model, data_generator, data_generator_wrapper

BASE_DIR = os.path.join(HOME_DIR, 'DLCV/Detection/yolo/keras-yolo3')

## 학습을 위한 기반 환경 설정. annotation 파일 위치, epochs시 저장된 모델 파일, Object클래스 파일, anchor 파일.
annotation_path = os.path.join(ANNO_DIR, 'raccoon_anno.csv')
log_dir = os.path.join(BASE_DIR, 'snapshots/000/')  # tensorboard, epoch 당 weight 저장 장소로 사용할 될 예정이다.
classes_path = os.path.join(BASE_DIR, 'model_data/raccoon_class.txt')    # 내가 만듬 한줄 - raccoon
# tiny yolo로 모델을 학습 원할 시 아래를 yolo_anchors.txt' -> tiny_yolo_anchors.txt'로 수정. 
anchors_path = os.path.join(BASE_DIR,'model_data/yolo_anchors.txt')      # 그대로 사용

class_names = get_classes(classes_path)
num_classes = len(class_names)   # 1개
anchors = get_anchors(anchors_path)

# 아래는 원본 train.py에서 weights_path 변경을 위해 임의 수정. 최초 weight 모델 로딩은 coco로 pretrained된 모델 로딩. 
# tiny yolo로 모델을 학습 원할 시 아래를 model_data/yolo.h5' -> model_data/tiny-yolo.h5'로 수정. 
model_weights_path = os.path.join(BASE_DIR, 'model_data/yolo.h5' )

input_shape = (416,416) # yolo-3 416 을 사용하므로 이렇게 정의. 라쿤 이미지들 wh는 모두 다르다.

is_tiny_version = len(anchors)==6 # default setting
# create_tiny_model(), create_model() 함수의 인자 설정을 원본 train.py에서 수정. 
if is_tiny_version:
    model = create_tiny_model(input_shape, anchors, num_classes,
        freeze_body=2, weights_path=model_weights_path)
else:
    # create_model 은 해당 패키지의 tarin.py 내부에 있는 클래스를 사용했다. 이 함수는 keras 모듈이 많이 사용한다. 우선 모르는 건 pass하고 넘어가자.
    model = create_model(input_shape, anchors, num_classes,
        freeze_body=2, weights_path=model_weights_path) # make sure you know what you freeze

# epoch 마다 call back 하여 모델 파일 저장.
# 이것 또한 Keras에서 많이 사용하는 checkpoint 저장 방식인듯 하다. 우선 이것도 모르지만 넘어가자.
logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

```

    Create YOLOv3 model with 9 anchors and 1 classes.
    Load weights /home/sb020518/DLCV/Detection/yolo/keras-yolo3/model_data/yolo.h5.
    Freeze the first 249 layers of total 252 layers.


### 3-3 전처리한 데이터로 학습시키기 - 학습과 검증 데이터로 나누고 학습!

- 학습 전에 GPU정리를 하자. 
- P100에서 40분정도 걸린다.
- 1 epoch당 weight가 저장되는 것을 확인할 수 있다.   
<p align="center"><img src='https://user-images.githubusercontent.com/46951365/92200652-6a695280-eeb5-11ea-86fa-8add3bbf7f05.png' alt='drawing' width='400'/></p>   
- 데이터가 적기 때문에, 최종 Loss값이 큰 것을 확인할 수 있다. 하지만 그래도 Inference 성능이 나쁘지 않다. Class 갯수가 1개 이므로..


```python
val_split = 0.1  # train data : val_data = 9 : 1

with open(annotation_path) as f:
    # 이러니 annotation 파일이 txt이든 csv이든 상관없었다.
    lines = f.readlines()

# 랜덤 시드 생성 및 lines 셔플하기
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)

# 데이터셋 나누기
num_val = int(len(lines)*val_split)
num_train = len(lines) - num_val

# 여기서 부터 진짜 학습 시작! 
# create_model() 로 반환된 yolo모델에서 trainable=False로 되어 있는 layer들 제외하고 학습
if True:
    # optimizer와 loss 함수 정의
    # 위에서 사용한 create_model 클래스의 맴버함수를 사용한다. 
    model.compile(optimizer=Adam(lr=1e-3), loss={
        # use custom yolo_loss Lambda layer.
        'yolo_loss': lambda y_true, y_pred: y_pred})

    batch_size = 2
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    
    # foward -> backpropagation -> weight 갱신 -> weight 저장
    # checkpoint 만드는 것은 뭔지 모르겠으니 pass...
    model.fit_generator(
            data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=50,
            initial_epoch=0,
            callbacks=[logging, checkpoint])
    model.save_weights(log_dir + 'trained_weights_stage_1.h5')

# create_model() 로 반환된 yolo모델에서 trainable=False로 되어 있는 layer들 없이, 모두 True로 만들고 다시 학습
if True:
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
    print('Unfreeze all of the layers.')

    batch_size = 4 # note that more GPU memory is required after unfreezing the body
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(
        data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, num_train//batch_size),
        validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
        validation_steps=max(1, num_val//batch_size),
        epochs=100,
        initial_epoch=50,
        callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    model.save_weights(log_dir + 'trained_weights_final.h5')
```

    Train on 180 samples, val on 20 samples, with batch size 2.
    Epoch 1/50
    90/90 [==============================] - 501s 6s/step - loss: 933.7634 - val_loss: 111.7494
    Epoch 2/50
    66/90 [=====================>........] - ETA: 1:58 - loss: 89.7261


    KeyboardInterrupt: 



```python
# YOLO 객체 생성. 
import sys
import argparse
#keras-yolo에서 image처리를 주요 PIL로 수행. 
from PIL import Image

LOCAL_PACKAGE_DIR = os.path.abspath("./keras-yolo3")
sys.path.append(LOCAL_PACKAGE_DIR)

```


```python
from yolo import YOLO, detect_video
raccoon_yolo = YOLO(
            model_path=os.path.join(HOME_DIR,'DLCV/Detection/yolo/keras-yolo3/snapshots/000/trained_weights_final.h5'),
            anchors_path='~/DLCV/Detection/yolo/keras-yolo3/model_data/yolo_anchors.txt',
            classes_path='~/DLCV/Detection/yolo/keras-yolo3/model_data/raccoon_class.txt')
```


```python
import matplotlib
import matplotlib.pyplot as plt

img = Image.open(os.path.join(IMAGE_DIR, 'raccoon-171.jpg'))

plt.figure(figsize=(12, 12))
plt.imshow(img)
```


```python
detected_img = raccoon_yolo.detect_image(img)

plt.figure(figsize=(12, 12))
plt.imshow(detected_img)
```

- 이제 모델구성까지 모두 끝났다. 이제 **raccoon_yolo.detect_image(img)** 만 잘사용하면 밑의 내용은 끝이다. 

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/92201281-1d867b80-eeb7-11ea-84b5-10c289d3e35e.png' alt='drawing' width='700'/></p>

### 3-4 임의의 16개의 원본 이미지를 추출하여 Object Detected된 결과 시각화 


```python
import numpy as np
np.random.seed(0)

# 랜덤하게 16개의 이미지 파일만 선택.
# 랜덤 숫자 만을 이용하지 말고 아래와 같은 방법을 이용하면, 알아서 random하게 파일을 선택해준다. 
all_image_files = glob.glob(IMAGE_DIR + '/*.jpg')
all_image_files = np.array(all_image_files)
file_cnt = all_image_files.shape[0]
show_cnt = 16
show_indexes = np.random.choice(file_cnt, show_cnt)
show_files = all_image_files[show_indexes]
print(show_files)

# 16개의 
fig, axs = plt.subplots(figsize=(24,24) , ncols=4 , nrows=4)
for i , filename in enumerate(show_files):
    print(filename)
    row = int(i/4)
    col = i%4
    img = Image.open(os.path.join(IMAGE_DIR, filename))
    detected_image = raccoon_yolo.detect_image(img)
    axs[row][col].imshow(detected_image)
    
```

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/92201468-9e457780-eeb7-11ea-8517-62d62294c431.png' alt='drawing' width='600'/></p>

# 4. 위의 내용들을 다시 활용 - Video Object Detection 수행. 
- video frame 가져오는 것은 cv2 사용. 
- 해당 프래임 이미지를 narray에서 PIL image로 변환해서, model.detect_image(image) 수행


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
detect_video_yolo(raccoon_yolo, '../../data/video/jack_and_raccoon.mp4', '../../data/output/jack_and_raccoon_yolo_01.avi')
```

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/92201594-efee0200-eeb7-11ea-9094-87610ca482a3.png' alt='drawing' width='600'/></p>

# 5. 위의 내용들을 반복 - tiny yolo로 학습. 

- 위에서 했던 내용을 통채로 함수로 만들자.


```python
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from train import get_classes, get_anchors
from train import create_model, create_tiny_model, data_generator, data_generator_wrapper

def train_yolo(pretrained_path, annotation_path,classes_path, anchors_path,log_dir,trained_model_name, b_size, epochs_cnt):      
        
        print('pretrained_path:', pretrained_path)
        class_names = get_classes(classes_path)
        num_classes = len(class_names)
        anchors = get_anchors(anchors_path)

        input_shape = (416,416) # multiple of 32, hw
        # tiny yolo여부를 anchor 설정 파일에서 자동으로 알 수 있음. anchor갯수가 6개이면 tiny yolo
        is_tiny_version = len(anchors)==6 # default setting
        
        # create_tiny_model(), create_model() 함수의 인자 설정을 원본 train.py에서 수정.
        if is_tiny_version:
            model = create_tiny_model(input_shape, anchors, num_classes,
                freeze_body=2, weights_path=pretrained_path)
        else:
            model = create_model(input_shape, anchors, num_classes,
                freeze_body=2, weights_path=pretrained_path) # make sure you know what you freeze

        logging = TensorBoard(log_dir=log_dir)
        checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
            monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

        val_split = 0.1
        with open(annotation_path) as f:
            lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_val = int(len(lines)*val_split)
        num_train = len(lines) - num_val

        # Train with frozen layers first, to get a stable loss.
        # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
        if True:
            model.compile(optimizer=Adam(lr=1e-3), loss={
                # use custom yolo_loss Lambda layer.
                'yolo_loss': lambda y_true, y_pred: y_pred})

            batch_size = b_size
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                    steps_per_epoch=max(1, num_train//batch_size),
                    validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                    validation_steps=max(1, num_val//batch_size),
                    epochs=int(epochs_cnt/2),
                    initial_epoch=0,
                    callbacks=[logging, checkpoint])
            model.save_weights(log_dir + trained_model_name+'_stage_1.h5')

        # Unfreeze and continue training, to fine-tune.
        # Train longer if the result is not good.
        if True:
            for i in range(len(model.layers)):
                model.layers[i].trainable = True
            model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
            print('Unfreeze all of the layers.')

            batch_size = b_size # note that more GPU memory is required after unfreezing the body
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=epochs_cnt,
                initial_epoch=int(epochs_cnt/2),
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])
            model.save_weights(log_dir + trained_model_name+'_final.h5')
```


```python
BASE_DIR = os.path.join(HOME_DIR, 'DLCV/Detection/yolo/keras-yolo3')
# keras-yolo3에서 convert 된 yolo-tiny pretrained 모델을 사용해야 함. 
pretrained_path = os.path.join(BASE_DIR, 'model_data/yolo-tiny.h5')
annotation_path = os.path.join(ANNO_DIR,'raccoon_anno.csv')
classes_path = os.path.join(BASE_DIR, 'model_data/raccoon_class.txt')
anchors_path = os.path.join(BASE_DIR, 'model_data/tiny_yolo_anchors.txt')
log_dir = os.path.join(BASE_DIR,'snapshots/000/')
trained_model_name = 'raccoon'
b_size=4
epochs_cnt = 100

train_yolo(pretrained_path, annotation_path,classes_path, anchors_path, log_dir,trained_model_name, b_size, epochs_cnt)
```


```python
raccoon_tiny_yolo = YOLO(model_path=os.path.join(HOME_DIR,'DLCV/Detection/yolo/keras-yolo3/snapshots/000/raccoon_final.h5'),
            anchors_path='~/DLCV/Detection/yolo/keras-yolo3/model_data/tiny_yolo_anchors.txt',
            classes_path='~/DLCV/Detection/yolo/keras-yolo3/model_data/raccoon_class.txt')
```


```python
# 16개의 이미지를 Detect해보자. 
import numpy as np
np.random.seed(0)

# 모든 이미지 파일중에서 임의의 16개 파일만 설정. 
all_image_files = glob.glob(IMAGE_DIR + '/*.jpg')
all_image_files = np.array(all_image_files)
file_cnt = all_image_files.shape[0]
show_cnt = 16

show_indexes = np.random.choice(file_cnt, show_cnt)
show_files = all_image_files[show_indexes]
print(show_files)
fig, axs = plt.subplots(figsize=(24,24) , ncols=4 , nrows=4)

for i , filename in enumerate(show_files):
    print(filename)
    row = int(i/4)
    col = i%4
    img = Image.open(os.path.join(IMAGE_DIR, filename))
    detected_image = raccoon_tiny_yolo.detect_image(img)
    axs[row][col].imshow(detected_image)
    
```


```python
detect_video_yolo(raccoon_tiny_yolo, '../../data/video/jack_and_raccoon.mp4', '../../data/output/jack_and_raccoon_tiny_yolo_01.avi')
```

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/92202191-650e0700-eeb9-11ea-9ab6-28367ea6d5a0.png' alt='drawing' width='600'/></p>

음... tiny yolo... 성능이 심각하게 안 좋다. 