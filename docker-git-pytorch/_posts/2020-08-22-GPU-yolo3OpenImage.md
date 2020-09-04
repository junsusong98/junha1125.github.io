---
layout: post
title: 【Keras】 Google Open Image 학습시키기 - Keras기반 Yolo3
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
- Keras-Yolo3에서 데이터를 활용하기 위한 변환 과정 (이미지 있는 코드 적극 활용 위해)
    1. OIDv4 Toolkit으로 데이터 다운로드
    2. VOC XML 타입으로 변경
    3. Keras yolo용 CSV로 변환
    

# 3. 데이터 다운로드 및 전처리 해보기