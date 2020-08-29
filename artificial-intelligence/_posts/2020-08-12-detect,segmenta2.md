---
layout: post
title: 【Vision】 Detection과 Segmentation 다시 정리 2 - Datasets
description: >
  당연하다고 생각하지만, 아직은 공부할게 많은 Detection과 Segmentation에 대한 개념을 다시 상기하고 정리해보면서 공부해볼 계획이다.
---

Detection과 Segmentation 다시 정리 2

# Dataset에 대한 진지한 고찰

## 1. Pascal VOC 기본개념

1. 기본 개념

   - XML Format (20개 Class)
   - 하나의 이미지에 대해서 Annotation이 하나의 XML 파일에 저장되어 있다.
   - [데이터셋 홈페이지 바로가기](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#data)
   - 대회 종류
     - Classification/Detection
     - Segmentation
     - Action Classification
     - Person Layout
   - Annotation : Pascal VOC 같은 경우에 아래와 같은 형식의 Annotation 파일로써 Ground True 정보를 파악할 수 있다.

   - Data Set을 다운받으면 다음과 같은 구조를 확인할 수 있다. segmented : segmentation 정보를 가지고 있느냐?를 의미한다.   
    <img src='https://user-images.githubusercontent.com/46951365/91629428-11587500-ea04-11ea-9319-20f4c16739c2.png' alt='drawing' width='500'/>  
    <img src='https://user-images.githubusercontent.com/46951365/91629436-1e756400-ea04-11ea-98fa-4f6e07487ac3.png' alt='drawing' width='500'/>  
   


## 2. PASCAL VOC 2012 데이터 탐색해보기  
  - 코드 안에 주석도 매우 중요하니 꼭 확인하고 공부하기
  - DLCV/Detection/preliminary/PASCAL_VOC_Dataset_탐색하기.ipynb 파일을 통해 공부한 내용

  ```python
  !ls ~/DLCV/data/voc/VOCdevkit/VOC2012
  ```

      Annotations  ImageSets	JPEGImages  SegmentationClass  SegmentationObject



  ```python
  !ls ~/DLCV/data/voc/VOCdevkit/VOC2012/JPEGImages | head -n 5
  ```

      2007_000027.jpg
      2007_000032.jpg
      2007_000033.jpg
      2007_000039.jpg
      2007_000042.jpg
      ls: write error: Broken pipe


  - JPEGImages 디렉토리에 있는 임의의 이미지 보기 


  ```python
  import cv2
  import matplotlib.pyplot as plt
  %matplotlib inline

  img = cv2.imread('../../data/voc/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg')
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  print('img shape:', img.shape)

  plt.figure(figsize=(8, 8))
  plt.imshow(img_rgb)
  plt.show()

  ```

      img shape: (281, 500, 3)


  - Annotations 디렉토리에 있는 임의의 annotation 파일 보기


  ```python
  !cat ~/DLCV/data/voc/VOCdevkit/VOC2012/Annotations/2007_000032.xml
  ```

      <annotation>
        <folder>VOC2012</folder>
        <filename>2007_000032.jpg</filename>
        <source>
          <database>The VOC2007 Database</database>
          <annotation>PASCAL VOC2007</annotation>
          <image>flickr</image>
        </source>
        <size>
          <width>500</width>
          <height>281</height>
          <depth>3</depth>
        </size>
        <segmented>1</segmented>
        <object>
          <name>aeroplane</name>
          <pose>Frontal</pose>
          <truncated>0</truncated>
          <difficult>0</difficult>
          <bndbox>
            <xmin>104</xmin>
            <ymin>78</ymin>
            <xmax>375</xmax>
            <ymax>183</ymax>
          </bndbox>
        </object>
        <object>
          <name>aeroplane</name>
          <pose>Left</pose>
          <truncated>0</truncated>
          <difficult>0</difficult>
          <bndbox>
            <xmin>133</xmin>
            <ymin>88</ymin>
            <xmax>197</xmax>
            <ymax>123</ymax>
          </bndbox>
        </object>
        <object>
          <name>person</name>
          <pose>Rear</pose>
          <truncated>0</truncated>
          <difficult>0</difficult>
          <bndbox>
            <xmin>195</xmin>
            <ymin>180</ymin>
            <xmax>213</xmax>
            <ymax>229</ymax>
          </bndbox>
        </object>
        <object>
          <name>person</name>
          <pose>Rear</pose>
          <truncated>0</truncated>
          <difficult>0</difficult>
          <bndbox>
            <xmin>26</xmin>
            <ymin>189</ymin>
            <xmax>44</xmax>
            <ymax>238</ymax>
          </bndbox>
        </object>
      </annotation>


  - SegmentationObject 디렉토리에 있는 있는 임의의 maksing 이미지 보기 


  ```python
  img = cv2.imread('../../data/voc/VOCdevkit/VOC2012/SegmentationObject/2007_000032.png')
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  print('img shape:', img.shape)

  plt.figure(figsize=(8, 8))
  plt.imshow(img_rgb)
  plt.show()
  ```

      img shape: (281, 500, 3)


  - Annotation xml 파일에 있는 요소들을 파싱하여 접근하기
  - 파일 이름만 뽑아아서 xml_files 라는 변수에 저장해 두기


  ```python
  # 파일 이름만 뽑아아서 xml_files 라는 변수에 저장해 두기
  import os
  import random

  VOC_ROOT_DIR ="../../data/voc/VOCdevkit/VOC2012/"
  ANNO_DIR = os.path.join(VOC_ROOT_DIR, "Annotations")
  IMAGE_DIR = os.path.join(VOC_ROOT_DIR, "JPEGImages")

  xml_files = os.listdir(ANNO_DIR)                       
  print(xml_files[:5]); print(len(xml_files))
  ```

      ['2012_001360.xml', '2008_003384.xml', '2008_007317.xml', '2009_000990.xml', '2009_003539.xml']
      17125


  - xml 파일을 다루기 위한 모듈은 다음과 같다. 


  ```python
  # !pip install lxml
  # 틔
  import os
  import xml.etree.ElementTree as ET

  xml_file = os.path.join(ANNO_DIR, '2007_000032.xml')

  # XML 파일을 Parsing 하여 Element 생성
  # 이렇게 2번의 과정을 거쳐서 Parsing을 완료하면 root라는 변수에 원하는 정보들이 저장되어 있다. 
  tree = ET.parse(xml_file)
  root = tree.getroot()

  # root가 dictionary 변수이면 root.keys()이렇게 출력하면 될덴데...
  print("root.keys = ", end='')
  for child in root:
      print(child.tag, end = ', ')
          

  # image 관련 정보는 root의 자식으로 존재
  # root를 이용해서 dictionary형식의 anotation 정보를 뽑아내는 방법은 다음과 같다.
  image_name = root.find('filename').text
  full_image_name = os.path.join(IMAGE_DIR, image_name)
  image_size = root.find('size')
  image_width = int(image_size.find('width').text)
  image_height = int(image_size.find('height').text)

  # 파일내에 있는 모든 object Element를 찾음.
  objects_list = []
  for obj in root.findall('object'):
      # object element의 자식 element에서 bndbox를 찾음. 
      xmlbox = obj.find('bndbox')
      # bndbox element의 자식 element에서 xmin,ymin,xmax,ymax를 찾고 이의 값(text)를 추출 
      x1 = int(xmlbox.find('xmin').text)
      y1 = int(xmlbox.find('ymin').text)
      x2 = int(xmlbox.find('xmax').text)
      y2 = int(xmlbox.find('ymax').text)
      
      bndbox_pos = (x1, y1, x2, y2)
      class_name=obj.find('name').text
      object_dict={'class_name': class_name, 'bndbox_pos':bndbox_pos}
      objects_list.append(object_dict)

  print('\nfull_image_name:', full_image_name,'\n', 'image_size:', (image_width, image_height))

  for object in objects_list:
      print(object)

      
  ```

      root.keys = folder, filename, source, size, segmented, object, object, object, object, 
      full_image_name: ../../data/voc/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg 
      image_size: (500, 281)
      {'class_name': 'aeroplane', 'bndbox_pos': (104, 78, 375, 183)}
      {'class_name': 'aeroplane', 'bndbox_pos': (133, 88, 197, 123)}
      {'class_name': 'person', 'bndbox_pos': (195, 180, 213, 229)}
      {'class_name': 'person', 'bndbox_pos': (26, 189, 44, 238)}


  - Annotation내의 Object들의 bounding box 정보를 이용하여 Bounding box 시각화


  ```python
  import cv2
  import os
  import xml.etree.ElementTree as ET

  xml_file = os.path.join(ANNO_DIR, '2007_000032.xml')

  tree = ET.parse(xml_file)
  root = tree.getroot()

  image_name = root.find('filename').text
  full_image_name = os.path.join(IMAGE_DIR, image_name)

  img = cv2.imread(full_image_name)
  # opencv의 rectangle()는 인자로 들어온 이미지 배열에 그대로 사각형을 그려주므로 별도의 이미지 배열에 그림 작업 수행. 
  draw_img = img.copy()
  # OpenCV는 RGB가 아니라 BGR이므로 빨간색은 (0, 0, 255)
  green_color=(0, 255, 0)
  red_color=(0, 0, 255)

  # 파일내에 있는 모든 object Element를 찾음.
  objects_list = []
  for obj in root.findall('object'):
      xmlbox = obj.find('bndbox')
      
      left = int(xmlbox.find('xmin').text)
      top = int(xmlbox.find('ymin').text)
      right = int(xmlbox.find('xmax').text)
      bottom = int(xmlbox.find('ymax').text)
      
      class_name=obj.find('name').text
      
      # draw_img 배열의 좌상단 우하단 좌표에 녹색으로 box 표시 
      cv2.rectangle(draw_img, (left, top), (right, bottom), color=green_color, thickness=1)
      # draw_img 배열의 좌상단 좌표에 빨간색으로 클래스명 표시
      cv2.putText(draw_img, class_name, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, red_color, thickness=1)

  img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
  plt.figure(figsize=(10, 10))
  plt.imshow(img_rgb)
  ```

  <img src='https://user-images.githubusercontent.com/46951365/91629962-7d3cdc80-ea08-11ea-81f3-db5e89f15715.png' alt='drawing' width='700'/>  




## 3. MS COCO

- json Format (80개 Class : paper에서는 80개라고 했지만, 이미지에서는 굳이 분류 안한 class가 있으니 주의할 것.)
- 300K 이미지들 1.5M개 Object들 (하나의 이미지당 5개의 객체)
- 모든 이미지에 대해서 하나의 json 파일이 존재한다.
- Pretrained Weight로 활용하기에 좋다.
- [데이터셋 홈페이지 바로가기](https://cocodataset.org/#download)
  - 보통 2017데이터 셋 가장 최신의 데이터셋을 사용한다. 
- [데이터셋 Explore를 쉽게 할 수 있다.](https://cocodataset.org/#explore)
- 데이터셋 구성  
    <img src='https://user-images.githubusercontent.com/46951365/91630158-18828180-ea0a-11ea-8a68-ddcf9449dfd8.png' alt='drawing' width='700'/>      
- 하나의 이미지 안에 여러 Class들 여러 Object들이 존재하고, 타 데이터 세트에 비해 난이도가 높은 데이터이다.
- 실제 우리의 환경에서 잘 동작할 수 있는 모델을 만들기 위해 만들어진 데이터셋이다. 

## 4. Google Open Image

- csv Format (600개 Class)
- Size도 매우 크기 때문에 학습에도 오랜 시간이 걸릴 수 있다.
