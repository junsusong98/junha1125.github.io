---
layout: post
title: 【Pytorch Package】SSD Pytorch Research
description: >
    Pytorch tuto를 모두 공부하고, 더 깊은 공부를 위해 SSD package를 공부한다.
---  

- SSD package를 가장 먼저 공부하는 이유는, SSD가 신경망 패키지 공부에 가장 쉽겠다고 생각했기 때문이다.
- 무엇으로 공부하지..? 라고 고민을 먼저 했다. 가장 유명한 detection 패키지인 Detectron2, mmdetection으로 공부해볼까 했지만, 그것보다 좀 더 쉬운 하나 detection 패키지만을 먼저 공부해보는게 좋겠다고 생각했다

# 0. Find Reference Github Code
1. amdegroot/ssd.pytorch
    - star : 4.1k, Contributors : 9, pytorch version : 1.0.0
    - last commit : 19.5 lnitial commit : 17.4
2. [sgrvinod/a-PyTorch-Tutorial-to-Object-Detection](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection)
    - star : 1.8k, Contributors : 2, pytorch version : 0.4
    - last commit : 20.8 lnitial commit : 18.8
    - Great Explanation about basics to SSD details
3. [**lufficc/SSD**](https://github.com/lufficc/SSD)
    - star : 1k, Contributors : 8, pytorch version : 1.4
    - last commit : 20.11 lnitial commit : 19.12
    - influenced by ssd.pytorch, aim to be the code base (very modular)

I will use lufficc/SSC repo. I think that this is up-to-date repository and developers emphasize this repo is high quality, fast and **modular**.

# 1. Analysis of lufficc/SSD
1. 독자적으로 사용하는 pypi패키지가 2개 있다.
    - yacs : config파일을 다루기 쉽게, CN()이라는 자체적은 클래스를 제작해놓음. 아래와 깉이 디렉토리 구조로 내용을 담고 있다.
        ```sh
        DATASETS:
            TEST: ()
            TRAIN: ()
        DATA_LOADER:
            NUM_WORKERS: 8
            PIN_MEMORY: True
        INPUT:
            IMAGE_SIZE: 300
            PIXEL_MEAN: [123, 117, 104]
        ```
    - vizer : 이미지에 vizualization을 해주는 툴을 직접 개발해 놓음. 이미지와 box좌표, 이미지와 mask좌표가 있다면 그걸로 이미지에 그림을 그려 시각화를 해줌.
2. VOC와 COCO data를 직접 다운
    - 과거에 정리해둔 [Datasets 정리 포스트](https://junha1125.github.io/artificial-intelligence/2020-08-12-detect,segmenta2/)를 보고 파일 다운로드
    - 그리고 export 설정으로 우분투 터미널에게 path를 알려줘야 한다. 
    - 경로 설정 까지 완료 (코랩 ssd_package_setup.ipynb 파일 참조) 
3. 그 후, 코드 공부 순서는 다음과 같다
    - Demo
        - $ python demo.py --config-file configs/vgg_ssd300_voc0712.yaml --images_dir demo --ckpt https://github.com/lufficc/SSD/releases/download/1.2/vgg_ssd300_voc0712.pth
    - Single GPU training
        - $ python train.py --config-file configs/vgg_ssd300_voc0712.yaml
    - Single GPU evaluating
        - $ python test.py --config-file configs/vgg_ssd300_voc0712.yaml


