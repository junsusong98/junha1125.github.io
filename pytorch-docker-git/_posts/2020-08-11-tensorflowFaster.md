---
layout: post
title: 【Tensorflow】Faster RCNN Inference 수행하기 + GPU 자원 주의사항
# description: > 

---

Tensorflow 1.3. Faster RCNN API로 Object Detection 수행하기  
/DLCV/Detection/fast_rcnn/Tensorflow_FasterRCNN_ObjectDetection.ipynb 참조

## 0. Tensorflow inferece 과정
1. 이미지 read 하기
2. .pb 파일만 읽어오기 - tf.gfile.FastGFile, graph_def = tf.GraphDef() 사용
3. 세션을 시작한다 - with tf.Session() as sess: 
4. 세션 내부에서 graph를 import한다 - tf.import_graph_def(graph_def, name='')
5. sess.run으로 forward처리하고, 원하는 정보를 뽑아온다. out = sess.run
6. 객체 하나하나에 대한 정보를 추출하여 시각화 한다 - for i in range(int(out[0][0])):


# 1. GPU 자원 주의사항

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/91733396-6ce15900-ebe4-11ea-8374-29d15df7b464.png' alt='drawing' width='600'/></p>

**해결방안**
- $ nvidia-smi 를 주기적으로 확인하고 학습을 시작하기 전에,   
    1. Jupyter - Running - 안쓰는 Notebook Shutdown 하기
    2. Notebook - Restart & Clear output 하기
    3. nvidia-smi 에서 나오는 process 중 GPU많이 사용하는 프로세서
    Kill -9 \<Processer ID>하기
    4. Jupyter Notebook 을 Terminal에서 kill하고 다시 키기 (~
    /start_nb.sh)


# 2. tensorflow로 Object Detection 수행하기 

## 1. 단일 이미지 Object Detection


```python
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

img = cv2.imread('../../data/image/john_wick01.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print('image shape:', img.shape)
plt.figure(figsize=(12, 12))
plt.imshow(img_rgb)
```

    image shape: (450, 814, 3)



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
```


```python
# !mkdir pretrained; cd pretrained
# !wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz
# !wget https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/faster_rcnn_resnet50_coco_2018_01_28.pbtxt
# cd faster_rcnn_resnet50_coco_2018_01_28; mv faster_rcnn_resnet50_coco_2018_01_28.pbtxt graph.pbtxt
```


```python
import numpy as np
import tensorflow as tf
import cv2
import time
import matplotlib.pyplot as plt
%matplotlib inline


#inference graph를 읽음. .
with tf.gfile.FastGFile('./pretrained/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
with tf.Session() as sess:
    # Session 시작하고 inference graph 모델 로딩 
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    # 여기서 session 내부에 graph가 들어가게 된다. 후에 sess변수를 사용하면서 grpah 정보를 가져올 수 있다. 
    
    # 입력 이미지 생성 및 BGR을 RGB로 변경 
    img = cv2.imread('../../data/image/beatles01.jpg')
    draw_img = img.copy()
    rows = img.shape[0]
    cols = img.shape[1]
    input_img = img[:, :, [2, 1, 0]]   # BGR -> RGB
    
    start = time.time()
    # Object Detection 수행. 
    # run - graph.get을 통해서 내가 가져오고 싶은 것을 인자로 적어놓는다. 순서대로 [객체수, 신뢰도, Box위치, Class]
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': input_img.reshape(1, input_img.shape[0], input_img.shape[1], 3) } ) # 이미지 여러게 
    print('type of out:', type(out), 'length of out:',len(out))  # list(4) = [객체수, 신뢰도, Box위치, Class]
    print(out)
    green_color=(0, 255, 0)
    red_color=(0, 0, 255)
    
    # Bounding Box 시각화 
    num_detections = int(out[0][0])
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > 0.5:
            left = bbox[1] * cols
            top = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            cv2.rectangle(draw_img, (int(left), int(top)), (int(right), int(bottom)), green_color, thickness=2)
            caption = "{}: {:.4f}".format(labels_to_names[classId], score)
            print(caption)
            cv2.putText(draw_img, caption, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, red_color, 1)
    
    print('Detection 수행시간:',round(time.time() - start, 2),"초")
    
img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 12))
plt.imshow(img_rgb)

# NMS 필터링에서 약간의 문제가 있는듯 하다... 약간 결과가 꺼림직하다. 
        
```

    type of out: <class 'list'> length of out: 4
    밑의 내용 : [객체수, 신뢰도, Box위치, Class]

    [array([19.], dtype=float32), 
    
    array([[0.99974984, 0.99930644, 0.9980475 , 0.9970795 , 0.9222008 ,
            0.8515703 , 0.8055376 , 0.7321974 , 0.7169089 , 0.6350252 ,
            0.6057731 , 0.5482028 , 0.51252437, 0.46408176, 0.43892667,
            0.41287616, 0.4075464 , 0.39610404, 0.3171757 , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            .
            ...
            .
            .

            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ]],
          dtype=float32), 
          
    array([[[0.40270284, 0.2723695 , 0.8693631 , 0.46764165],
            [0.40439418, 0.06080557, 0.88185936, 0.24013077],
            [0.40899867, 0.68438506, 0.9282361 , 0.9033634 ],
            [0.42774147, 0.4751278 , 0.8887425 , 0.7367553 ],
            [0.3681334 , 0.5855469 , 0.41420895, 0.6274197 ],
            [0.36090973, 0.7612593 , 0.46531847, 0.78825235],
            [0.35362682, 0.5422665 , 0.3779468 , 0.56790847],
            [0.35872525, 0.47497243, 0.37832502, 0.4952262 ],
            [0.39067298, 0.17564818, 0.54261357, 0.31135702],
            [0.3596046 , 0.6206162 , 0.4659364 , 0.7180736 ],
            [0.36052787, 0.7542875 , 0.45949724, 0.7803741 ],
            [0.35740715, 0.55126834, 0.38326728, 0.57657194],
            [0.36718863, 0.5769864 , 0.40654665, 0.61239254],
            [0.35574582, 0.4798463 , 0.37322614, 0.4985193 ],
            [0.35036406, 0.5329462 , 0.3708444 , 0.5514975 ],
            [0.367587  , 0.39456058, 0.41583234, 0.43441534],
            [0.3562084 , 0.47724184, 0.37217227, 0.49240994],
            [0.36195153, 0.6252996 , 0.46575055, 0.72400415],
            [0.36365557, 0.5674811 , 0.39475283, 0.59136254],
            [0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        ],
            ...
            .
            .
            ...
            .
            [0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        ]]], dtype=float32), 
            
            
        array([[1., 1., 1., 1., 3., 1., 3., 3., 3., 8., 1., 3., 3., 3., 3., 3.,
            3., 3., 3., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.]], dtype=float32)]

    person: 0.9997
    person: 0.9993
    person: 0.9980
    person: 0.9971
    car: 0.9222
    person: 0.8516
    car: 0.8055
    car: 0.7322
    car: 0.7169
    truck: 0.6350
    person: 0.6058
    car: 0.5482
    car: 0.5125

    Detection 수행시간: 12.99 초

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/91736659-fdba3380-ebe8-11ea-91c2-64e3b9a7fad6.png' alt='drawing' width='600'/></p>


## 2. 위의 과정 함수로 def하고 활용해보기


```python
def get_tensor_detected_image(sess, img_array, use_copied_array):
    
    rows = img_array.shape[0]
    cols = img_array.shape[1]
    if use_copied_array:
        draw_img_array = img_array.copy()
    else:
        draw_img_array = img_array
    
    input_img = img_array[:, :, [2, 1, 0]]  # BGR2RGB

    start = time.time()
    # Object Detection 수행. 
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': input_img.reshape(1, input_img.shape[0], input_img.shape[1], 3)})
    
    green_color=(0, 255, 0)
    red_color=(0, 0, 255)
    
    # Bounding Box 시각화 
    num_detections = int(out[0][0])
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > 0.5:
            left = bbox[1] * cols
            top = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            cv2.rectangle(draw_img_array, (int(left), int(top)), (int(right), int(bottom)), green_color, thickness=2)
            caption = "{}: {:.4f}".format(labels_to_names[classId], score)
            cv2.putText(draw_img_array, caption, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, red_color, 1)
            #print(caption)
    print('Detection 수행시간:',round(time.time() - start, 2),"초")
    return draw_img_array
# end of function

```

방금 위에서 만든 함수 사용해서 Image Object Detection 수행하기 
{:.lead}


```python
with tf.gfile.FastGFile('./pretrained/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
with tf.Session() as sess:
    # Session 시작하고 inference graph 모델 로딩 
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    
    # 입력 이미지 생성 및 BGR을 RGB로 변경 
    img = cv2.imread('../../data/image/john_wick01.jpg')
    draw_img = get_tensor_detected_image(sess, img, True)

img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 12))
plt.imshow(img_rgb)
```

    Detection 수행시간: 15.58 초

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/91736647-f85ce900-ebe8-11ea-902d-dceb9c76ddf1.png' alt='drawing' width='600'/></p>




## 3. 위에서 만든 함수로 Video Object Detection 수행


```python
video_input_path = '../../data/video/John_Wick_small.mp4'
# linux에서 video output의 확장자는 반드시 avi 로 설정 필요. 
video_output_path = '../../data/output/John_Wick_small_tensor01.avi'

cap = cv2.VideoCapture(video_input_path)

codec = cv2.VideoWriter_fourcc(*'XVID')

vid_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
vid_fps = cap.get(cv2.CAP_PROP_FPS)
    
vid_writer = cv2.VideoWriter(video_output_path, codec, vid_fps, vid_size) 

frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('총 Frame 갯수:', frame_cnt)

green_color=(0, 255, 0)
red_color=(0, 0, 255)

#inference graph를 읽음. .
with tf.gfile.FastGFile('./pretrained/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
with tf.Session() as sess:
    # Session 시작하고 inference graph 모델 로딩 
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

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
!gsutil cp ../../data/output/John_Wick_small_tensor01.avi gs://my_bucket_dlcv/data/output/John_Wick_small_tensor01.avi
```

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/91717662-e8371080-ebcc-11ea-845d-09796062d2e5.png' alt='drawing' width='600'/></p>

    

