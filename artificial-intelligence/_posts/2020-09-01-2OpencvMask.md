---
layout: post
title: 【Python-Module】 Mask-RCNN 수행하기 - OpenCV DNN 모듈
description: >
     이전 Post를 통해서 Mask-RCNN의 이론에 대해서 공부했다. OpenCV DNN 모델을 사용해서 Mask-RCNN를 수행해보자. 
---

Mask-RCNN 수행하기 - OpenCV DNN 모듈  
OpenCV 사용과정 요약은 [이전 Post](https://junha1125.github.io/artificial-intelligence/2020-08-15-2OpenCVcode/) 참조  

주의 사항
{:.lead}
1. 지금까지와의 과정은 다르다. Segmentation 정보도 가져와야 한다.   
따라서 그냥 forward()하면 안되고, foward(['detection_out_final', 'detection_masks'])를 해줘야 한다.  
2. Network에서 예측된 Mask정보를 원본 이미지 비율로 확장해야한다. 
3. 시각화를 해야하는데, 그냥 원본 이미지에 mask정보를 덮어버리면 안되고, **약간 투명하게 mask정보를 시각화**해야한다. 
4. conda activate tf113 환경 사용하지
4. /home/sb020518/DLCV/Segmentation/mask_rcnn 파일 참고할 것
5. 여기에서는 cv2사용하는 것 말고, 전처리 및 데이터 사용에 대해서 깊게 살펴보기

# 0. OpenCV 파일 다운로드

- Tensorflow에서 Pretrained 된 Inference모델(Frozen graph)와 환경파일을 다운로드 받은 후 이를 이용해 OpenCV에서 Inference 모델 생성
* https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API 에 다운로드 URL 있음.
* pretrained 모델은 http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz 에서 다운로드 후 압축 해제
* pretrained 모델을 위한 환경 파일은 https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt 에서 다운로드. 하고 이름은 graph.pbtxt 로 바꿔주기
* download된 모델 파일과 config 파일을 인자로 하여 inference 모델을 DNN에서 로딩함. 



```python
import cv2
import matplotlib.pyplot as plt
import numpy  as np
%matplotlib inline

img = cv2.imread('../../data/image/driving.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 12))
plt.axis('off')
plt.imshow(img_rgb)

```



<p align="center"><img src='https://user-images.githubusercontent.com/46951365/93009985-17348580-f5c2-11ea-849b-b2d0eb2c5888.png' alt='drawing' width='600'/></p>

- Tensorflow의 Object Detection 모델 Weight인 frozen_inference_graph와 graph.pbtxt를 
- 이용하여 Opencv 의 dnn Network 모델로 로딩
- 이전에는 forward()만을 사용했지만 이제는 **boxes, masks = cv_net.forward(['detection_out_final', 'detection_masks'])**
- 과거 yolo를 사용했을때, cv_outs = cv_net_yolo.forward(outlayer_names) 를 사용했다. 
- forward([,])에 들어가야하는 인자는 layer name이 되어야 한다!!

# 1. 신경망 cv_net 구상하기


```python
cv_net = cv2.dnn.readNetFromTensorflow('./pretrained/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb', 
                                     './pretrained/mask_rcnn_inception_v2_coco_2018_01_28/graph.pbtxt')

blob = cv2.dnn.blobFromImage(img , swapRB=True, crop=False)
cv_net.setInput(blob)

# Bounding box 정보는 detection_out_final layer에서 mask 정보는 detection_masks layer에서 추출. 
boxes, masks = cv_net.forward(['detection_out_final', 'detection_masks'])
```


```python
layer_names = cv_net.getLayerNames()
outlayer_names = [layer_names[i[0] - 1] for i in cv_net.getUnconnectedOutLayers()]
print(type(layer_names),len(layer_names))
print(layer_names.index('detection_out_final'), layer_names.index('detection_masks'))
```

    <class 'list'> 332
    260 331



```python
# coco dataset의 클래스 ID별 클래스명 매핑
labels_to_names_seq= {0:'person',1:'bicycle',2:'car',3:'motorcycle',4:'airplane',5:'bus',6:'train',7:'truck',8:'boat',9:'traffic light',
                    10:'fire hydrant',11:'street sign',12:'stop sign',13:'parking meter',14:'bench',15:'bird',16:'cat',17:'dog',18:'horse',19:'sheep',
                    20:'cow',21:'elephant',22:'bear',23:'zebra',24:'giraffe',25:'hat',26:'backpack',27:'umbrella',28:'shoe',29:'eye glasses',
                    30:'handbag',31:'tie',32:'suitcase',33:'frisbee',34:'skis',35:'snowboard',36:'sports ball',37:'kite',38:'baseball bat',39:'baseball glove',
                    40:'skateboard',41:'surfboard',42:'tennis racket',43:'bottle',44:'plate',45:'wine glass',46:'cup',47:'fork',48:'knife',49:'spoon',
                    50:'bowl',51:'banana',52:'apple',53:'sandwich',54:'orange',55:'broccoli',56:'carrot',57:'hot dog',58:'pizza',59:'donut',
                    60:'cake',61:'chair',62:'couch',63:'potted plant',64:'bed',65:'mirror',66:'dining table',67:'window',68:'desk',69:'toilet',
                    70:'door',71:'tv',72:'laptop',73:'mouse',74:'remote',75:'keyboard',76:'cell phone',77:'microwave',78:'oven',79:'toaster',
                    80:'sink',81:'refrigerator',82:'blender',83:'book',84:'clock',85:'vase',86:'scissors',87:'teddy bear',88:'hair drier',89:'toothbrush',
                    90:'hair brush'}



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


```python
print('boxes shape:', boxes.shape, 'masks shape:', masks.shape)
# 각각이 무슨 역할인지는 바로 다음 코드 참조
# 지금 100개의 객체를 찾았다. confidence를 이용해서 몇개 거르긴 해야 함
```

    boxes shape: (1, 1, 100, 7) masks shape: (100, 90, 15, 15)


# 2. 하나의 객체 Detect 정보 시각화 하기(꼭 알아두기)
- 발견된 첫번째 객체에 대해서만, mask정보 추출 및 전처리
    <p align="center"><img src='https://user-images.githubusercontent.com/46951365/93011370-a5b00380-f5d0-11ea-8b35-594ce5d094cb.png' alt='drawing' width='500'/></p>


```python
numClasses = masks.shape[1]
numDetections = boxes.shape[2]

# opencv의 rectangle(), putText() API는 인자로 들어온 IMAGE array에 그대로 수정작업을 수행하므로 bounding box 적용을 위한 
# 별도의 image array 생성. 
draw_img = img.copy()

img_height = draw_img.shape[0]
img_width = draw_img.shape[1]

conf_threshold = 0.5
mask_threshold = 0.3

green_color=(0, 255, 0)
red_color=(0, 0, 255)

# 이미지를 mask 설명을 위해서 iteration을 한번만 수행. 
#for i in range(numDetections):
for i in range(1):
    box = boxes[0, 0, i] # box변수의 7개 원소를 가져온다.
    mask = masks[i]      # 90, 15, 15 shaep의 list. 90 : 각 클래스에 대해서 분류 이미지 depth, 15 x 15를 해당 객체 box size로 확장해야함
    score = box[2]       # 아하 box변수의 3번째 원소가 score값
    if score > conf_threshold:
        classId = int(box[1])             
        left = int(img_width * box[3])  # 정규화 되어 있으니까, w,h랑 곱해주는거 잊지말기
        top = int(img_height * box[4])
        right = int(img_width * box[5])
        bottom = int(img_height * box[6])
        
        # 이미지에 bounding box 그림 그리기
        text = "{}: {:.4f}".format(labels_to_names_seq[classId], score)
        cv2.rectangle(draw_img, (left, top), (right, bottom), green_color, thickness=2 )
        cv2.putText(draw_img, text, (left, top-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, red_color, 1)

        # mask 정보 처리하기
        classMask = mask[classId] # (15, 15)
        # 15 x 15 를 해당 객체 box size로 확장해야함
        scaled_classMask = cv2.resize(classMask, (right - left + 1, bottom - top + 1)) # (123, 224)
        # 지정된 mask Threshold 값 이상인지 True, False boolean형태의 mask 정보 생성. 
        s_mask_b = (scaled_classMask > mask_threshold) # bool형태의 (123, 224) 리스트 생성
        print('scaled mask shape:', s_mask_b.shape, '\nscaled mask pixel count:', s_mask_b.shape[0]*s_mask_b.shape[1],
              '\nscaled mask true shape:',s_mask_b[s_mask_b==True].shape, 
              '\nscaled mask False shape:', s_mask_b[s_mask_b==False].shape)
        # 원본 이미지의 bounding box 영역만 image 추출
        # 참고로 copy()해서 가져온게 아니기 때문에, before_mask_roi를 바꾸면 draw_img도 바뀐다
        before_mask_roi = draw_img[top:bottom+1, left:right+1]
        print('before_mask_roi:', before_mask_roi.shape)
```

    scaled mask shape: (123, 224) 
    scaled mask pixel count: 27552 
    scaled mask true shape: (22390,) 
    scaled mask False shape: (5162,)
    before_mask_roi: (123, 224, 3)


- **위 코드에서 얻은 첫번째 객체에 대한 마스크 정보를 이용해 시각화 준비.**


```python
vis_mask = (s_mask_b * 255).astype("uint8")  # true자리는 255로 만들어 준다

# mask정보를 덮어주는 작업을, cv2.bitwise_and 모듈을 사용해서 쉽게 한다.
# 객체가 있는 true자리만 그대로 놔두고, 아니면 검은색으로 바꿔버린다. vis_mask의 255가 검은색이다.
instance = cv2.bitwise_and(before_mask_roi, before_mask_roi, mask=vis_mask)

```


```python
fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=(8, 8), ncols=4, nrows=1)

ax1.set_title('network detected mask')
ax1.axis('off')
ax1.imshow(classMask)

ax2.set_title('resized mask')
ax2.axis('off')
ax2.imshow(scaled_classMask)


ax3.set_title('Before Mask ROI')
ax3.axis('off')
ax3.imshow(before_mask_roi)

ax4.set_title('after Mask ROI')
ax4.axis('off')
ax4.imshow(instance)
```

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/93010571-0f2c1400-f5c9-11ea-93d8-76f40c0f4ace.png' alt='drawing' width='700'/></p>

- **Detected된 object에 mask를 특정 투명 컬러로 적용후 시각화**
- 엄청 신기하게 하니까 알아두기 
- 이 코드를 보고 이해 안되면 최대한 이해해 보기
 <p align="center"><img src='https://user-images.githubusercontent.com/46951365/93010445-9e382c80-f5c7-11ea-9129-486b06dc23aa.png' alt='drawing' width='500'/></p>


```python
# 색갈을 하나 랜덤하게 골라서
colorIndex = np.random.randint(0, len(colors)-1)
color = colors[colorIndex]
after_mask_roi = draw_img[top:bottom+1, left:right+1][s_mask_b] # 필요한 부분만 때온다. 주의!! 2차원이 1차원으로 바뀐다! (depth정보는 그대로)
print(after_mask_roi.shape) 
# 이 코드를 계속 실행하면, 색을 덮고덮고덮고덮고를 반복한다. 랜덤으로 색이 바뀌는게 아니라...
draw_img[top:bottom+1, left:right+1][s_mask_b] = ([0.1*color[0], 0.1*color[1], 0.1*color[2]] + (0.5 * after_mask_roi)).astype(np.uint8)
# 2차원이든 1차원이든 쨋든 원하는 부분만 색을 바꿔준다. 투명 윈리는 위와 같다.

plt.figure(figsize=(6,6))
plt.axis('off')
plt.title('After Mask color')
plt.imshow(draw_img[top:bottom+1, left:right+1])
# 시각화 되는 건 draw_img의 일부분이지만, python variable immutable 때문에 draw_img의 전체도 바뀌어 있다

```

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/93010555-f4599f80-f5c8-11ea-96e9-39017be81b79.png' alt='drawing' width='400'/></p>

- 굳이 할 필요 없지만, Detect된 Object에 **contour 윤곽선 적용.** 
- cv2.findContours, cv2.drawContours모듈을 사용하면 쉽다. 


```python
s_mask_i = s_mask_b.astype(np.uint8)
# https://datascienceschool.net/view-notebook/f9f8983941254a34bf0fee42c66c5539/ 에 이미지 컨투어 설명 있음 
contours, hierarchy = cv2.findContours(s_mask_i,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(draw_img[top:bottom+1, left:right+1], contours, -1, color, 1, cv2.LINE_8, hierarchy, 100)

plt.figure(figsize=(6,6))
plt.axis('off')
plt.title('After Mask color')
plt.imshow(draw_img[top:bottom+1, left:right+1])
```

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/93010598-439fd000-f5c9-11ea-944e-14834feb4479.png' alt='drawing' width='500'/></p>

# 3. 모든 객체에 대해서 mask정보 그려주기
- 위에서 했던 작업 동일하게 반복 수행


```python
numClasses = masks.shape[1]
numDetections = boxes.shape[2]

# opencv의 rectangle(), putText() API는 인자로 들어온 IMAGE array에 그대로 수정작업을 수행하므로 bounding box 적용을 위한 
# 별도의 image array 생성. 
draw_img = img.copy()

img_height = draw_img.shape[0]
img_width = draw_img.shape[1]
conf_threshold = 0.5
mask_threshold = 0.3

green_color=(0, 255, 0)
red_color=(0, 0, 255)

for i in range(numDetections):
    box = boxes[0, 0, i]
    mask = masks[i]
    score = box[2]
    if score > conf_threshold:
        classId = int(box[1])
        left = int(img_width * box[3])
        top = int(img_height * box[4])
        right = int(img_width * box[5])
        bottom = int(img_height * box[6])

        text = "{}: {:.4f}".format(labels_to_names_seq[classId], score)
        cv2.rectangle(draw_img, (left, top), (right, bottom), green_color, thickness=2 )
        cv2.putText(draw_img, text, (left, top-3), cv2.FONT_HERSHEY_SIMPLEX, 2, red_color, 1)

        classMask = mask[classId]
        scaled_classMask = cv2.resize(classMask, (right - left + 1, bottom - top + 1))
        s_mask_b = (scaled_classMask > mask_threshold)
        
        # 마스크 정보 투명하게 색 덮어 주기
        before_mask_roi = draw_img[top:bottom+1, left:right+1]
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


<p align="center"><img src='https://user-images.githubusercontent.com/46951365/93010644-bc9f2780-f5c9-11ea-9fe3-5f624c307149.png' alt='drawing' width='800'/></p>

# 4. 바로 위에서 한 작업. 함수화 및 이미지에 함수 사용


```python
def get_box_info(box, img_width, img_height):
    
    classId = int(box[1])
    left = int(img_width * box[3])
    top = int(img_height * box[4])
    right = int(img_width * box[5])
    bottom = int(img_height * box[6])
    
    left = max(0, min(left, img_width - 1))
    top = max(0, min(top, img_height - 1))
    right = max(0, min(right, img_width - 1))
    bottom = max(0, min(bottom, img_height - 1))
    
    return classId, left, top, right, bottom

    
def draw_box(img_array, box, img_width, img_height, is_print=False):
    green_color=(0, 255, 0)
    red_color=(0, 0, 255)
    
    score = box[2]
    classId, left, top, right, bottom = get_box_info(box, img_width, img_height)
    text = "{}: {:.4f}".format(labels_to_names_seq[classId], score)
    
    if is_print:
        print("box:", box, "score:", score, "classId:", classId)
    
    cv2.rectangle(img_array, (left, top), (right, bottom), green_color, thickness=2 )
    cv2.putText(img_array, text, (left, top-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, thickness=1)
    
    return img_array
    
def draw_mask(img_array, box, mask, img_width, img_height, mask_threshold, is_print=False):
        
        classId, left, top, right, bottom = get_box_info(box, img_width, img_height)
        classMask = mask[classId]
        
        # 원본 이미지의 object 크기에 맞춰 mask 크기 scale out 
        scaled_classMask = cv2.resize(classMask, (right - left + 1, bottom - top + 1))
        s_mask_b = (scaled_classMask > mask_threshold)
        before_mask_roi = img_array[top:bottom+1, left:right+1]
        
        # mask를 적용할 bounding box 영역의 image 추출하고 투명 color 적용. 
        colorIndex = np.random.randint(0, len(colors)-1)
        color = colors[colorIndex]
        after_mask_roi = img_array[top:bottom+1, left:right+1][s_mask_b]
        img_array[top:bottom+1, left:right+1][s_mask_b] = ([0.3*color[0], 0.3*color[1], 0.3*color[2]] + 0.6 * after_mask_roi).astype(np.uint8)
        
        # Detect된 Object에 윤곽선(contour) 적용. 
        s_mask_i = s_mask_b.astype(np.uint8)
        contours, hierarchy = cv2.findContours(s_mask_i,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_array[top:bottom+1, left:right+1], contours, -1, color, 1, cv2.LINE_8, hierarchy, 100)
        
        return img_array
```

- 바로 위에서 만든 함수 사용해서, detect_image_mask_rcnn 함수 만들기


```python
import time

def detect_image_mask_rcnn(cv_net, img_array, conf_threshold, mask_threshold, use_copied_array, is_print=False):
    
    draw_img = None
    if use_copied_array:
        draw_img = img_array.copy()
        #draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
    else:
        draw_img = img_array
        
    start_time = time.time()
    
    blob = cv2.dnn.blobFromImage(img_array, swapRB=True, crop=False)
    cv_net.setInput(blob)
    boxes, masks = cv_net.forward(['detection_out_final', 'detection_masks'])
    
    inference_time = time.time() - start_time
    if is_print:
        print('Segmentation Inference time {0:}'.format(inference_time))

    numClasses = masks.shape[1]
    numDetections = boxes.shape[2]

    img_height = img_array.shape[0]
    img_width = img_array.shape[1]
    
    for i in range(numDetections):
        box = boxes[0, 0, i]
        mask = masks[i]
        score = box[2]
        #print("score:", score)
        if score > conf_threshold:
            draw_box(img_array , box, img_width, img_height, is_print=is_print)
            draw_mask(img_array, box, mask, img_width, img_height, mask_threshold, is_print=is_print)
    
    return img_array
```


```python
labels_to_names_seq = {0:'person',1:'bicycle',2:'car',3:'motorcycle',4:'airplane',5:'bus',6:'train',7:'truck',8:'boat',9:'traffic light',
                    10:'fire hydrant',11:'street sign',12:'stop sign',13:'parking meter',14:'bench',15:'bird',16:'cat',17:'dog',18:'horse',19:'sheep',
                    20:'cow',21:'elephant',22:'bear',23:'zebra',24:'giraffe',25:'hat',26:'backpack',27:'umbrella',28:'shoe',29:'eye glasses',
                    30:'handbag',31:'tie',32:'suitcase',33:'frisbee',34:'skis',35:'snowboard',36:'sports ball',37:'kite',38:'baseball bat',39:'baseball glove',
                    40:'skateboard',41:'surfboard',42:'tennis racket',43:'bottle',44:'plate',45:'wine glass',46:'cup',47:'fork',48:'knife',49:'spoon',
                    50:'bowl',51:'banana',52:'apple',53:'sandwich',54:'orange',55:'broccoli',56:'carrot',57:'hot dog',58:'pizza',59:'donut',
                    60:'cake',61:'chair',62:'couch',63:'potted plant',64:'bed',65:'mirror',66:'dining table',67:'window',68:'desk',69:'toilet',
                    70:'door',71:'tv',72:'laptop',73:'mouse',74:'remote',75:'keyboard',76:'cell phone',77:'microwave',78:'oven',79:'toaster',
                    80:'sink',81:'refrigerator',82:'blender',83:'book',84:'clock',85:'vase',86:'scissors',87:'teddy bear',88:'hair drier',89:'toothbrush',
                    90:'hair brush'}
```

- 위에서 만든 함수를 사용해서 instance segmentation 수행하기


```python
img = cv2.imread('../../data/image/baseball01.jpg')

cv_net = cv2.dnn.readNetFromTensorflow('./pretrained/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb', 
                                     './pretrained/mask_rcnn_inception_v2_coco_2018_01_28/graph.pbtxt')

img_detected = detect_image_mask_rcnn(cv_net, img, conf_threshold=0.5, mask_threshold=0.3, use_copied_array=True, is_print=True)

img_rgb = cv2.cvtColor(img_detected, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 12))
plt.imshow(img_rgb)

```

- 위에서 만든 함수를 사용해서 instance segmentation 수행하기


```python
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

wick_img = cv2.imread('../../data/image/john_wick01.jpg')

wick_img_detected = detect_image_mask_rcnn(cv_net, wick_img, conf_threshold=0.5, mask_threshold=0.3, use_copied_array=True, is_print=True)

wick_img_rgb = cv2.cvtColor(wick_img_detected, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 12))
plt.imshow(wick_img_rgb)
```


<p align="center"><img src='https://user-images.githubusercontent.com/46951365/93010693-261f3600-f5ca-11ea-9e65-5d17245e520a.png' alt='drawing' width='700'/></p>

# 5. 동영상에 Segmentation 적용. 위에서 만든 함수 사용


```python
def detect_video_mask_rcnn(cv_net, input_path, output_path, conf_threshold, mask_threshold,  is_print):
    
    cap = cv2.VideoCapture(input_path)

    codec = cv2.VideoWriter_fourcc(*'XVID')

    vid_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)

    vid_writer = cv2.VideoWriter(output_path, codec, 24, vid_size) 

    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('총 Frame 갯수:', frame_cnt, )

    frame_index=0
    while True:
        hasFrame, img_frame = cap.read()
        frame_index += 1
        if not hasFrame:
            print('더 이상 처리할 frame이 없습니다.')
            break
        print("frame index:{0:}".format(frame_index), end=" ")
        returned_frame = detect_image_mask_rcnn(cv_net, img_frame, conf_threshold=conf_threshold,
                                                mask_threshold=mask_threshold,use_copied_array=False, is_print=is_print)
        vid_writer.write(returned_frame)
    # end of while loop

    vid_writer.release()
    cap.release()
```


```python
cv_net = cv2.dnn.readNetFromTensorflow('./pretrained/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb', 
                                     './pretrained/mask_rcnn_inception_v2_coco_2018_01_28/graph.pbtxt')

detect_video_mask_rcnn(cv_net, '../../data/video/John_Wick_small.mp4', '../../data/output/John_Wick_mask_01.avi',
                      conf_threshold=0.5, mask_threshold=0.3, is_print=True)
```


```python
!gsutil cp ../../data/output/John_Wick_mask_01.avi gs://my_bucket_dlcv/data/output/John_Wick_mask_01.avi
```



