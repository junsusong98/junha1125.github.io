---
layout: post
title: 【Tensorflow】Mask-RCNN inference 수행하기
# description: > 

---

Tensorflow v1.3 API로, Mask-RCNN inference 수행하기  
/DLCV/Segmentation/mask_rcnn/Tensorflow를_이용한_Mask_RCNN_Segmentation.ipynb 참고하기   
[이번 post](https://junha1125.github.io/docker-git-pytorch/2020-08-11-tensorflowFaster/)를 참고해서, Tensorflofw inference 흐름 파악하기

# 1. 이미지 Read 

- Tensorflow에서 Pretrained 된 Inference모델(Frozen graph)을 다운로드 받은 후 이를 이용해 OpenCV에서 Inference 모델 생성
- [이전 post](https://junha1125.github.io/artificial-intelligence/2020-09-01-2OpencvMask/)에서 다운 받은 .pb 파일 그대로 사용할 예정
    * https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API 에 다운로드 URL 있음.


```python
import cv2
import matplotlib.pyplot as plt
import numpy  as np
%matplotlib inline

beatles_img = cv2.imread('../../data/image/driving2.jpg')
beatles_img_rgb = cv2.cvtColor(beatles_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 12))
plt.imshow(beatles_img_rgb)

```

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/93011693-50c1bc80-f5d3-11ea-88cd-8b4d9758137f.png' alt='drawing' width='600'/></p>


```python
# coco dataset의 클래스 ID별 클래스명 매핑
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


#masking 시 클래스별 컬러 적용
colors = list(
    [[0, 255, 0],
     [0, 0, 255],
     [255, 0, 0],
     [0, 255, 255],
     [255, 255, 0],
     [255, 0, 255],
     [80, 70, 180],
     [250, 80, 190],
     [245, 145, 50],
     [70, 150, 250],
     [50, 190, 190]] )
```

# 2. import graph & 한 이미지 Inference 
- tensorflow inference 수행하기
- 아래의 내용이 이해가 알될 수도 있다. 그러면 꼭 이전 [post(OpenCV-Maskrcnn)](https://junha1125.github.io/artificial-intelligence/2020-09-01-2OpencvMask/) 참고하기
- **주의! sess.run 할 때, 이전과 다르게 graph.get_tensor_by_name('detection_masks:0')도 가져와야한다.**


```python
import numpy as np
import tensorflow as tf
import cv2
import time
import matplotlib.pyplot as plt
%matplotlib inline


#inference graph를 읽음. .
with tf.gfile.FastGFile('./pretrained/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
with tf.Session() as sess:
    # Session 시작하고 inference graph 모델 로딩 
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    
    # 입력 이미지 생성 및 BGR을 RGB로 변경 
    img = cv2.imread('../../data/image/driving2.jpg')
    draw_img = img.copy()
    
    img_height = img.shape[0]
    img_width = img.shape[1]
    #inp = cv2.resize(img, (300, 300))
    # OpenCV로 입력받은 BGR 이미지를 RGB 이미지로 변환 
    inp = img[:, :, [2, 1, 0]] 

    start = time.time()
    # Object Detection 수행 및 mask 정보 추출. 'detection_masks:0' 에서 mask결과 추출 
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0'),
                    sess.graph.get_tensor_by_name('detection_masks:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
    
    green_color=(0, 255, 0)
    red_color=(0, 0, 255)
    score_threshold = 0.5
    mask_threshold = 0.4
    
    #### out 결과, 타입 Debugging #### 
    print("### out 크기와 타입:", len(out), type(out))
    print(out[0].shape, out[1].shape, out[2].shape, out[3].shape, out[4].shape)
    print('num_detection:',out[0], 'score by objects:', out[1][0], 'bounding box')
    # Bounding Box 시각화 
    # Detect된 Object 별로 bounding box 시각화 
    num_detections = int(out[0][0])
    for i in range(num_detections):
        # Object별 class id와 object class score, bounding box정보를 추출
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        # Object별 mask 정보 추출
        classMask = out[4][0][i]
        
        if score > score_threshold:
            left = int(bbox[1] * img_width)
            top = int(bbox[0] * img_height)
            right = int(bbox[3] * img_width)
            bottom = int(bbox[2] * img_height)
            # cv2의 rectangle(), putText()로 bounding box의 클래스명 시각화 
            cv2.rectangle(draw_img, (left, top), (right,bottom ), green_color, thickness=2)
            caption = "{}: {:.4f}".format(labels_to_names[classId], score)
            print(caption)
            cv2.putText(draw_img, caption, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, red_color, 1)
            
            # 원본 이미지의 object 크기에 맞춰 mask 크기 scale out 
            scaled_classMask = cv2.resize(classMask, (right - left + 1, bottom - top + 1))
            print('원본 이미지 비율로 scale out된 classMask shape:', scaled_classMask.shape)
            # 지정된 mask Threshold 값 이상인지 True, False boolean형태의 mask 정보 생성. 
            s_mask_b = (scaled_classMask > mask_threshold)
            print('scaled mask shape:', s_mask_b.shape, 'scaled mask pixel count:', s_mask_b.shape[0]*s_mask_b.shape[1],
                  'scaled mask true shape:',s_mask_b[s_mask_b==True].shape, 
                  'scaled mask False shape:', s_mask_b[s_mask_b==False].shape)
            
            # mask를 적용할 bounding box 영역의 image 추출
            before_mask_roi = draw_img[top:bottom+1, left:right+1]
            print('before_mask_roi:', before_mask_roi.shape)
            
            
            # Detect된 Object에 mask를 특정 투명 컬러로 적용. 
            colorIndex = np.random.randint(0, len(colors)-1)
            color = colors[colorIndex]
            after_mask_roi = draw_img[top:bottom+1, left:right+1][s_mask_b]
            draw_img[top:bottom+1, left:right+1][s_mask_b] = ([0.3*color[0], 0.3*color[1], 0.3*color[2]] + 0.6 * after_mask_roi).astype(np.uint8)
            # Detect된 Object에 윤곽선(contour) 적용. 
            s_mask_i = s_mask_b.astype(np.uint8)
            contours, hierarchy = cv2.findContours(s_mask_i,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(draw_img[top:bottom+1, left:right+1], contours, -1, color, 1, cv2.LINE_8, hierarchy, 100)

plt.figure(figsize=(14, 14))
img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()
```

```sh
car: 0.9624
원본 이미지 비율로 scale out된 classMask shape: (414, 510)
scaled mask shape: (414, 510) scaled mask pixel count: 211140 scaled mask true shape: (155967,) scaled mask False shape: (55173,)
before_mask_roi: (414, 510, 3)
car: 0.9441
원본 이미지 비율로 scale out된 classMask shape: (225, 251)
scaled mask shape: (225, 251) scaled mask pixel count: 56475 scaled mask true shape: (40707,) scaled mask False shape: (15768,)
before_mask_roi: (225, 251, 3)
traffic light: 0.9239
원본 이미지 비율로 scale out된 classMask shape: (91, 50)
scaled mask shape: (91, 50) scaled mask pixel count: 4550 scaled mask true shape: (4003,) scaled mask False shape: (547,)
before_mask_roi: (91, 50, 3)
car: 0.9119
원본 이미지 비율로 scale out된 classMask shape: (153, 183)
scaled mask shape: (153, 183) scaled mask pixel count: 27999 scaled mask true shape: (21834,) scaled mask False shape: (6165,)
before_mask_roi: (153, 183, 3)
car: 0.8999
원본 이미지 비율로 scale out된 classMask shape: (67, 94)
scaled mask shape: (67, 94) scaled mask pixel count: 6298 scaled mask true shape: (5041,) scaled mask False shape: (1257,)
before_mask_roi: (67, 94, 3)
```
<p align="center"><img src='https://user-images.githubusercontent.com/46951365/93011764-27edf700-f5d4-11ea-8ab3-64d63cd1beef.png' alt='drawing' width='600'/></p>

    

# 3. 위의 작업 함수화 및 함수 사용


```python
## 이전 opencv에서 선언한 get_box_info()함수에 left, top, right, bottom을 가지오는 bbox 위치 인덱스 변경
def get_box_info(bbox, img_width, img_height):
    
    left = int(bbox[1] * img_width)
    top = int(bbox[0] * img_height)
    right = int(bbox[3] * img_width)
    bottom = int(bbox[2] * img_height)
    
    left = max(0, min(left, img_width - 1))
    top = max(0, min(top, img_height - 1))
    right = max(0, min(right, img_width - 1))
    bottom = max(0, min(bottom, img_height - 1))
    
    return left, top, right, bottom

# 이전 opencv에서 선언한 draw_box()함수에 classId, score 인자가 추가됨.     
def draw_box(img_array, classId, score, box, img_width, img_height, is_print=False):
    green_color=(0, 255, 0)
    red_color=(0, 0, 255)
    
    left, top, right, bottom = get_box_info(box, img_width, img_height)
    text = "{}: {:.4f}".format(labels_to_names[classId], score)
    
    if is_print:
        pass
        #print("box:", box, "score:", score, "classId:", classId)
    
    cv2.rectangle(img_array, (left, top), (right, bottom), green_color, thickness=2 )
    cv2.putText(img_array, text, (left, top-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, thickness=1)
    
    return img_array
    
def draw_mask(img_array, bbox, classMask, img_width, img_height, mask_threshold, is_print=False):
        
        left, top, right, bottom = get_box_info(bbox, img_width, img_height)
        # 원본 이미지의 object 크기에 맞춰 mask 크기 scale out 
        scaled_classMask = cv2.resize(classMask, (right - left + 1, bottom - top + 1))
        s_mask_b = (scaled_classMask > mask_threshold)
        before_mask_roi = img_array[top:bottom+1, left:right+1]
        
        # mask를 적용할 bounding box 영역의 image 추출하고 투명 color 적용. 
        colorIndex = np.random.randint(0, len(colors)-1)
        #color = colors[colorIndex]
        color=(224, 32, 180)
        after_mask_roi = img_array[top:bottom+1, left:right+1][s_mask_b]
        img_array[top:bottom+1, left:right+1][s_mask_b] = ([0.3*color[0], 0.3*color[1], 0.3*color[2]] + 0.6 * after_mask_roi).astype(np.uint8)
        # Detect된 Object에 윤곽선(contour) 적용. 
        s_mask_i = s_mask_b.astype(np.uint8)
        contours, hierarchy = cv2.findContours(s_mask_i,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_array[top:bottom+1, left:right+1], contours, -1, color, 1, cv2.LINE_8, hierarchy, 100)
        
        return img_array
```


```python
import time

def detect_image_mask_rcnn_tensor(sess, img_array, conf_threshold, mask_threshold, use_copied_array, is_print=False):
    
    draw_img = None
    if use_copied_array:
        draw_img = img_array.copy()
        #draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
    else:
        draw_img = img_array
    
    img_height = img_array.shape[0]
    img_width = img_array.shape[1]
    
    # BGR을 RGB로 변환하여 INPUT IMAGE 입력 준비
    inp = img_array[:, :, [2, 1, 0]]  

    start = time.time()
    # Object Detection 수행 및 mask 정보 추출. 'detection_masks:0' 에서 mask결과 추출 
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0'),
                    sess.graph.get_tensor_by_name('detection_masks:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
    if is_print:
        print('Segmentation Inference time {0:}'.format(round(time.time() - start, 4)))
        
    num_detections = int(out[0][0])

    for i in range(num_detections):
        # Object별 class id와 object class score, bounding box정보를 추출
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        # Object별 mask 정보 추출
        classMask = out[4][0][i]

        if score > conf_threshold:
            draw_box(img_array , classId, score, bbox, img_width, img_height, is_print=is_print)
            draw_mask(img_array, bbox, classMask, img_width, img_height, mask_threshold, is_print=is_print)
    
    return img_array
```


```python
#inference graph를 읽음. .
with tf.gfile.FastGFile('./pretrained/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
with tf.Session() as sess:
    # Session 시작하고 inference graph 모델 로딩 
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    
    # 입력 이미지 생성, Object Detection된 image 반환, 반환된 image의 BGR을 RGB로 변경 
    img = cv2.imread('../../data/image/beatles01.jpg')
    draw_img = detect_image_mask_rcnn_tensor(sess, img, conf_threshold=0.5, mask_threshold=0.4, use_copied_array=True, is_print=True)

img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 12))
plt.imshow(img_rgb)
```

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/93011780-4d7b0080-f5d4-11ea-8da5-c04e59b606d9.png' alt='drawing' width='600'/></p>

# 4. 위에서 만든 함수 사용해서, Video Segmentation 적용
- 10fps 정도의 좋은 속도가 나오는 것을 확인할 수 있다.
- 아래에 sess.close() 가 있다. 이것은 사실이 코드가 먼저 실행 되어야 한다. 이 코드는 sess을 with로 실행 하지 않음.  
    <img src='https://user-images.githubusercontent.com/46951365/93011853-2a9d1c00-f5d5-11ea-9e05-2d75441c5fb6.png' alt='drawing' width='700'/>


```python
from IPython.display import clear_output, Image, display, Video, HTML
Video('../../data/video/London_Street.mp4')
```


```python
video_input_path = '../../data/video/London_Street.mp4'
# linux에서 video output의 확장자는 반드시 avi 로 설정 필요. 
video_output_path = '../../data/output/London_Street_mask_rcnn_01.avi'

cap = cv2.VideoCapture(video_input_path)

codec = cv2.VideoWriter_fourcc(*'XVID')

vid_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
vid_fps = cap.get(cv2.CAP_PROP_FPS)
    
vid_writer = cv2.VideoWriter(video_output_path, codec, vid_fps, vid_size) 

frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('총 Frame 갯수:', frame_cnt, 'FPS:', vid_fps )

frame_index = 0
while True:
    hasFrame, img_frame = cap.read()
    if not hasFrame:
        print('더 이상 처리할 frame이 없습니다.')
        break
    frame_index += 1
    print("frame index:{0:}".format(frame_index), end=" ")
    draw_img_frame = detect_image_mask_rcnn_tensor(sess, img_frame, conf_threshold=0.5, mask_threshold=0.4, use_copied_array=False, is_print=True)
    vid_writer.write(draw_img_frame)
# end of while loop

vid_writer.release()
cap.release()  

```


```python
!gsutil cp ../../data/output/London_Street_mask_rcnn_01.avi gs://my_bucket_dlcv/data/output/London_Street_mask_rcnn_01.avi
```


```python
sess.close()  # 
```

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/93011830-e4e05380-f5d4-11ea-81a2-1a19c9875ce2.png' alt='drawing' width='600'/></p>
