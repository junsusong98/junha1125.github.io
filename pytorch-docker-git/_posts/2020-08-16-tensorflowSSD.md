---
layout: post
title: 【Tensorflow】SSD Inference 수행하기
# description: > 

---

Tensorflow v1.3 API로, SSD Inference 수행하기     

/DLCV/Detection/ssd/Tensorflow_SSD_이미지와_영상_Detection.ipynb 참조

/DLCV/Detection/ssd/SSD_Face_Detection.md

[이번 post](https://junha1125.github.io/docker-git-pytorch/2020-08-11-tensorflowFaster/)를 참고해서, Tensorflofw inference 흐름 파악하기

## 1. Tensorflow로 SSD - Object Detection 수행하기 
- [이전 Post](https://junha1125.github.io/artificial-intelligence/2020-08-16-SSD2OpenCV/)에서 받은,     
  download 받은 SSD+Inception, SSD+Mobilenet 그래프 모델을 이용하여 Tensorflow에서 이미지와 비디오 Object Detectio 수행

- SSD+Inception, SSD+Mobilenet의 config파일은 각각 /DLCV/Detection/ssd/pretrained/   "graph.pbtxt",   "graph2.pbtxt" 이다.  
    하지만 여기서는 graph.pbtxt와 graph2.pbtxt를 사용할 필요가 없다. pb 파일만 필요하다^^


```python
#### ms coco 클래스 아이디와 클래스 명 매핑
labels_to_names = {1:'person',2:'bicycle',3:'car',4:'motorcycle',5:'airplane',6:'bus',7:'train',8:'truck',9:'boat',10:'traffic light',
                    11:'fire hydrant',12:'street sign',13:'stop sign',14:'parking meter',15:'bench',16:'bird',17:'cat',18:'dog',19:'horse',20:'sheep',
                    21:'cow',22:'elephant',23:'bear',24:'zebra',25:'giraffe',26:'hat',27:'backpack',28:'umbrella',29:'shoe',30:'eye glasses',
                    31:'handbag',32:'tie',33:'suitcase',34:'frisbee',35:'skis',36:'snowboard',37:'sports ball',38:'kite',39:'baseball bat',40:'baseball glove',
                    41:'skateboard',42:'surfboard',43:'tennis racket',44:'bottle',45:'plate',46:'wine glass',47:'cup',48:'fork',49:'knife',50:'spoon',
                    51:'bowl',52:'banana',53:'apple',54:'sandwich',55:'orange',56:'broccoli',57:'carrot',58:'hot dog',59:'pizza',60:'donut',
                    61:'cake',62:'chair',63:'couch',64:'potted plant',65:'bed',66:'mirror',67:'dining table',68:'window',69:'desk',70:'toilet',
                    71:'door',72:'tv',73:'laptop',74:'mouse',75:'remote',76:'keyboard',77:'cell phone',78:'microwave',79:'oven',80:'toaster',
                    81:'sink',82:'refrigerator',83:'blender',84:'book',85:'clock',86:'vase',87:'scissors',88:'teddy bear',89:'hair drier',90:'toothbrush',
                    91:'hair brush'}
```

### 1-1 SSD + Inception으로 단일 이미지 Object Detection


```python
import numpy as np
import tensorflow as tf
import cv2
import time
import matplotlib.pyplot as plt
%matplotlib inline


#inference graph를 읽음. .
with tf.gfile.FastGFile('./pretrained/ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
with tf.Session() as sess:
    # Session 시작하고 inference graph 모델 로딩 
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    
    # 입력 이미지 생성 및 BGR을 RGB로 변경 
    img = cv2.imread('../../data/image/beatles01.jpg')
    draw_img = img.copy()
    rows = img.shape[0]
    cols = img.shape[1]
    inp = cv2.resize(img, (300, 300))
    # OpenCV로 입력받은 BGR 이미지를 RGB 이미지로 변환 
    inp = inp[:, :, [2, 1, 0]] 

    start = time.time()
    # Object Detection 수행. 
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
    
    green_color=(0, 255, 0)
    red_color=(0, 0, 255)
    
    # Bounding Box 시각화 
    # Detect된 Object 별로 bounding box 시각화 
    num_detections = int(out[0][0])
    for i in range(num_detections):
        # class id와 object class score, bounding box정보를 추출
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > 0.4:
            left = bbox[1] * cols
            top = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            # cv2의 rectangle(), putText()로 bounding box의 클래스명 시각화 
            cv2.rectangle(draw_img, (int(left), int(top)), (int(right), int(bottom)), green_color, thickness=2)
            caption = "{}: {:.4f}".format(labels_to_names[classId], score)
            print(caption)
            cv2.putText(draw_img, caption, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, red_color, 1)
    
    print('Detection 수행시간:',round(time.time() - start, 2),"초")
    
img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 12))
plt.imshow(img_rgb)
        
```

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/91717571-b8880880-ebcc-11ea-81b4-601784dacfa6.png' alt='drawing' width='500'/></p>

### 1-2 위에서 한작업 함수로 def 하기. Session 내부에 있는 일부 작업만 def한 것!


```python
def get_tensor_detected_image(sess, img_array, use_copied_array):
    
    rows = img_array.shape[0]
    cols = img_array.shape[1]
    if use_copied_array:
        draw_img = img_array.copy()
    else:
        draw_img = img_array
    
    inp = cv2.resize(img_array, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
    
    start = time.time()
    # Object Detection 수행. 
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
    
    green_color=(0, 255, 0)
    red_color=(0, 0, 255)
    
    # Bounding Box 시각화 
    # Detect된 Object 별로 bounding box 시각화 
    num_detections = int(out[0][0])
    for i in range(num_detections):
        # class id와 object class score, bounding box정보를 추출
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > 0.4:
            left = bbox[1] * cols
            top = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            # cv2의 rectangle(), putText()로 bounding box의 클래스명 시각화 
            cv2.rectangle(draw_img, (int(left), int(top)), (int(right), int(bottom)), green_color, thickness=2)
            caption = "{}: {:.4f}".format(labels_to_names[classId], score)
            cv2.putText(draw_img, caption, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, red_color, 1)
    
    print('Detection 수행시간:',round(time.time() - start, 3),"초")
    return draw_img

```

### 1-3 위에서 만든 함수 사용해서 Inference 수행하기


```python
#inference graph를 읽음. .
with tf.gfile.FastGFile('./pretrained/ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
with tf.Session() as sess:
    # Session 시작하고 inference graph 모델 로딩 
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    
    # 입력 이미지 생성, Object Detection된 image 반환, 반환된 image의 BGR을 RGB로 변경 
    img = cv2.imread('../../data/image/beatles01.jpg')
    draw_img = get_tensor_detected_image(sess, img, True)

img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 12))
plt.imshow(img_rgb)
```

## 2. tensorflow로 SSD+ Inception 기반 video Object Detection 수행
- 위에서 만든 함수 사용


```python
video_input_path = '../../data/video/Night_Day_Chase.mp4'
# linux에서 video output의 확장자는 반드시 avi 로 설정 필요. 
video_output_path = '../../data/output/Night_Day_Chase_tensor_ssd_incep01.avi'

cap = cv2.VideoCapture(video_input_path)

codec = cv2.VideoWriter_fourcc(*'XVID')

vid_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
vid_fps = cap.get(cv2.CAP_PROP_FPS)
    
vid_writer = cv2.VideoWriter(video_output_path, codec, vid_fps, vid_size) 

frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('총 Frame 갯수:', frame_cnt, 'FPS:', vid_fps )

green_color=(0, 255, 0)
red_color=(0, 0, 255)

# SSD+Inception inference graph를 읽음. .
with tf.gfile.FastGFile('./pretrained/ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
with tf.Session() as sess:
    # Session 시작하고 inference graph 모델 로딩 
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    index = 0
    while True:
        hasFrame, img_frame = cap.read()
        if not hasFrame:
            print('더 이상 처리할 frame이 없습니다.')
            break

        draw_img_frame = get_tensor_detected_image(sess, img_frame, False)
        vid_writer.write(draw_img_frame)
    # end of while loop

vid_writer.release()
cap.release()  

```

- **SSD가 CPU에서 매우 빠르고 잘 되는 것을 확인할 수 있다.**
{:.lead}

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/91840851-51318d80-ec8c-11ea-9253-d7ec08a7e21d.png' alt='drawing' width='600'/></p>



```python
!gsutil cp ../../data/output/Night_Day_Chase_tensor_ssd_incep01.avi gs://my_bucket_dlcv/data/output/Night_Day_Chase_tensor_ssd_incep01.avi
```

## 3. SSD+Mobilenet Video Object Detection
- 아래 코드는 커널을 재 기동후 바로 수행해야 합니다. 
- Graph를 가져오는 것에 대해서 위에서 했던 것과 중첩되면 오류가 날 수 있다.
- inceptionV2 과 성능차이가 없지만, 수행시간이 CPU에서 더더 빠른 것을 확인할 수 있다. 

### 3-1 함수정의하기

```python
labels_to_names = {1:'person',2:'bicycle',3:'car',4:'motorcycle',5:'airplane',6:'bus',7:'train',8:'truck',9:'boat',10:'traffic light',
                    11:'fire hydrant',12:'street sign',13:'stop sign',14:'parking meter',15:'bench',16:'bird',17:'cat',18:'dog',19:'horse',20:'sheep',
                    21:'cow',22:'elephant',23:'bear',24:'zebra',25:'giraffe',26:'hat',27:'backpack',28:'umbrella',29:'shoe',30:'eye glasses',
                    31:'handbag',32:'tie',33:'suitcase',34:'frisbee',35:'skis',36:'snowboard',37:'sports ball',38:'kite',39:'baseball bat',40:'baseball glove',
                    41:'skateboard',42:'surfboard',43:'tennis racket',44:'bottle',45:'plate',46:'wine glass',47:'cup',48:'fork',49:'knife',50:'spoon',
                    51:'bowl',52:'banana',53:'apple',54:'sandwich',55:'orange',56:'broccoli',57:'carrot',58:'hot dog',59:'pizza',60:'donut',
                    61:'cake',62:'chair',63:'couch',64:'potted plant',65:'bed',66:'mirror',67:'dining table',68:'window',69:'desk',70:'toilet',
                    71:'door',72:'tv',73:'laptop',74:'mouse',75:'remote',76:'keyboard',77:'cell phone',78:'microwave',79:'oven',80:'toaster',
                    81:'sink',82:'refrigerator',83:'blender',84:'book',85:'clock',86:'vase',87:'scissors',88:'teddy bear',89:'hair drier',90:'toothbrush',
                    91:'hair brush'}

def get_tensor_detected_image(sess, img_array, use_copied_array):
    
    rows = img_array.shape[0]
    cols = img_array.shape[1]
    if use_copied_array:
        draw_img = img_array.copy()
    else:
        draw_img = img_array
    
    inp = cv2.resize(img_array, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
    

    start = time.time()
    # Object Detection 수행. 
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
    
    green_color=(0, 255, 0)
    red_color=(0, 0, 255)
    
    # Bounding Box 시각화 
    # Detect된 Object 별로 bounding box 시각화 
    num_detections = int(out[0][0])
    for i in range(num_detections):
        # class id와 object class score, bounding box정보를 추출
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > 0.3:
            left = bbox[1] * cols
            top = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            # cv2의 rectangle(), putText()로 bounding box의 클래스명 시각화 
            cv2.rectangle(draw_img, (int(left), int(top)), (int(right), int(bottom)), green_color, thickness=2)
            caption = "{}: {:.4f}".format(labels_to_names[classId], score)
            cv2.putText(draw_img, caption, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, red_color, 1)
    
    print('Detection 수행시간:',round(time.time() - start, 3),"초")
    return draw_img
```

### 3-2 정의한 함수사용해서 비디오 Detection 수행하기

```python
import numpy as np
import tensorflow as tf
import cv2
import time
import matplotlib.pyplot as plt
%matplotlib inline

video_input_path = '../../data/video/Night_Day_Chase.mp4'
# linux에서 video output의 확장자는 반드시 avi 로 설정 필요. 
video_output_path = '../../data/output/Night_Day_Chase_tensor_ssd_mobile_01.avi'

cap = cv2.VideoCapture(video_input_path)

codec = cv2.VideoWriter_fourcc(*'XVID')

vid_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
vid_fps = cap.get(cv2.CAP_PROP_FPS)
    
vid_writer = cv2.VideoWriter(video_output_path, codec, vid_fps, vid_size) 

frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('총 Frame 갯수:', frame_cnt, 'FPS:', vid_fps )

green_color=(0, 255, 0)
red_color=(0, 0, 255)

# SSD+Mobilenet inference graph를 읽음. .
with tf.gfile.FastGFile('./pretrained/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
with tf.Session() as sess:
    # Session 시작하고 inference graph 모델 로딩 
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    index = 0
    while True:
        hasFrame, img_frame = cap.read()
        if not hasFrame:
            print('더 이상 처리할 frame이 없습니다.')
            break

        draw_img_frame = get_tensor_detected_image(sess, img_frame, False)
        vid_writer.write(draw_img_frame)
    # end of while loop

vid_writer.release()
cap.release()  

```


```python
!gsutil cp ../../data/output/Night_Day_Chase_tensor_ssd_mobile_01.avi gs://my_bucket_dlcv/data/output/Night_Day_Chase_tensor_ssd_mobile_01.avi
```


## 4. Tensorflow Inference.pb 파일가져와 사용하기(Face Detection)

-  WiderFace 데이터세트로 Pretrained된 Tensorflow graph 모델을 다운로드 받아 이를 이용해 Face Detection 수행. 

- Github : https://github.com/yeephycho/tensorflow-face-detection

- SSD + MobileNet기반으로 Pretrained된 모델 다운로드 

- https://github.com/yeephycho/tensorflow-face-detection/raw/master/model/frozen_inference_graph_face.pb

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/91841699-c356a200-ec8d-11ea-9fa2-f6986f608a11.png' alt='drawing' width='400'/></p>

### 4-1 SSD + Mobilenet Pretrained된 모델 로딩하여 Face  Detection

```python
import numpy as np
import tensorflow as tf
import cv2
import time
import matplotlib.pyplot as plt
%matplotlib inline
        
```




```python
def get_tensor_detected_image(sess, img_array, use_copied_array):
    
    rows = img_array.shape[0]
    cols = img_array.shape[1]
    if use_copied_array:
        draw_img = img_array.copy()
    else:
        draw_img = img_array
    
    inp = cv2.resize(img_array, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
    
    start = time.time()
    # Object Detection 수행. 
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
    
    green_color=(0, 255, 0)
    red_color=(0, 0, 255)
    
    # Bounding Box 시각화 
    # Detect된 Object 별로 bounding box 시각화 
    num_detections = int(out[0][0])
    for i in range(num_detections):
        # class id와 object class score, bounding box정보를 추출
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > 0.4:
            left = bbox[1] * cols
            top = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            # cv2의 rectangle(), putText()로 bounding box의 클래스명 시각화 
            cv2.rectangle(draw_img, (int(left), int(top)), (int(right), int(bottom)), green_color, thickness=2)
            caption = "face: {:.4f}".format(score)
            cv2.putText(draw_img, caption, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 2)
    
    print('Detection 수행시간:',round(time.time() - start, 3),"초")
    return draw_img

#inference graph를 읽음. .
with tf.gfile.FastGFile('./pretrained/frozen_inference_graph_face.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
with tf.Session() as sess:
    # Session 시작하고 inference graph 모델 로딩 
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    
    # 입력 이미지 생성, Object Detection된 image 반환, 반환된 image의 BGR을 RGB로 변경 
    img = cv2.imread('../../data/image/EPL01.jpg')
    draw_img = get_tensor_detected_image(sess, img, True)

img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 12))
plt.imshow(img_rgb)
```

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/91845613-64e0f200-ec94-11ea-914a-a3f41390da44.png' alt='drawing' width='500'/></p>


### 4-2 tensorflow로 SSD+ Inception 기반 video Object Detection 수행


```python
from IPython.display import clear_output, Image, display, Video, HTML
Video('../../data/video/InfiniteWar01.mp4')
```



```python
video_input_path = '../../data/video/InfiniteWar01.mp4'
# linux에서 video output의 확장자는 반드시 avi 로 설정 필요. 
video_output_path = '../../data/output/InfiniteWar01_ssd.avi'

cap = cv2.VideoCapture(video_input_path)

codec = cv2.VideoWriter_fourcc(*'XVID')

vid_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
vid_fps = cap.get(cv2.CAP_PROP_FPS)
    
vid_writer = cv2.VideoWriter(video_output_path, codec, vid_fps, vid_size) 

frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('총 Frame 갯수:', frame_cnt, 'FPS:', vid_fps )

green_color=(0, 255, 0)
red_color=(0, 0, 255)

# SSD+Inception inference graph를 읽음. .
with tf.gfile.FastGFile('./pretrained/frozen_inference_graph_face.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
with tf.Session() as sess:
    # Session 시작하고 inference graph 모델 로딩 
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    index = 0
    while True:
        hasFrame, img_frame = cap.read()
        if not hasFrame:
            print('더 이상 처리할 frame이 없습니다.')
            break

        draw_img_frame = get_tensor_detected_image(sess, img_frame, False)
        vid_writer.write(draw_img_frame)
    # end of while loop

vid_writer.release()
cap.release()  

```

  
    Detection 수행시간: 0.307 초
    Detection 수행시간: 0.294 초
    Detection 수행시간: 0.296 초
    Detection 수행시간: 0.304 초
    Detection 수행시간: 0.296 초
    Detection 수행시간: 0.294 초
    Detection 수행시간: 0.293 초
    Detection 수행시간: 0.31 초
    더 이상 처리할 frame이 없습니다.



```python
!gsutil cp ../../data/output/InfiniteWar01_ssd.avi gs://my_bucket_dlcv/data/output/InfiniteWar01_ssd.avi
```

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/91845378-00be2e00-ec94-11ea-85d3-e39b94496449.png' alt='drawing' width='700'/></p>

