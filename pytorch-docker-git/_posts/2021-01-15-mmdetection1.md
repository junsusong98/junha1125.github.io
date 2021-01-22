---
layout: post
title: 【mmdetection】mmdetection SSD 사용해보며 전체 분석하기

---

- [mmdetection Github](https://github.com/open-mmlab/mmdetection) 
- 이전에 [detectron2와 mmdetection](https://junha1125.github.io/docker-git-pytorch/2021-01-08-SSD_pytorch/#detectron2--mmdetection-short-research)에 대해서 간단하게 분석해본 내용도 참고해보자. 



# 0. Readme 정리

1. [mmdetection/docs/model zoo](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md) 정리

   - **mmdetection/configs** : '다양한 종류의 신경망' 모델 설계를 위한, model_config<u>.py</u> 파일이 존재한다.

   - 각 '신경망 모델'이름의 폴더에 들어가면, readme.md가 따로 있고, 그곳에 <u>backbone, **style(pytorch/caffe 두가지 framework 사용됨)**, lr-schd, memory, fps, boxAP, cong, Download(model/log)</u> 가 적혀 있어서 참고하면 된다. 

2. installation([mmdetection/docs/get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md))

   - Prerequisites
     - PyTorch 1.3 to 1.6
     - [MMCV](https://mmcv.readthedocs.io/en/latest/#installation) 
   - installation
     - install mmcv : 이미 빌드가 된 버전 [다운로드](https://download.openmmlab.com/mmcv/dist/index.html) & Git을 다운받아 setup을 실행해 직접 빌드하거나.
     - install mmdetection : Only Github download + 직접 setup & build
     - note - 코드 수정은 reinstall이 필요없이, 바로 적용될 것 이다. 
     - 혹시 docker를 사용하고 싶다면, dockerfile도 사용해보기

3.  Getting Started (**아래의 내용들 빠르게 공부하자**)

    1.  [mmdetection/demo/MMDet_Tutorial.ipynb](https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_Tutorial.ipynb)
    2.  [mmdetection/docs/with existing dataset](https://github.com/open-mmlab/mmdetection/blob/master/docs/1_exist_data_model.md) 
    3.  [mmdetection/docs/with new dataset](https://github.com/open-mmlab/mmdetection/blob/master/docs/2_new_data_model.md) 
    4.  [mmdetection/demo/webcam_demo.py](https://github.com/open-mmlab/mmdetection/blob/master/demo/webcam_demo.py) for beginners
    5.  [mmdetection official documetation](https://mmdetection.readthedocs.io/en/latest/index.html) : 여기에도 좋은 내용이 많다. 3번까지 공부하고 5번의 내용과 이 documentation의 내용 중 더 맘에 드는 내용을 공부해보자.
    6.  (여기부터는 필요하면 공부하자. 내가 나중에 어떤 패키지를 가져와서 코드를 수정할지 모르니..)There are also tutorials for [finetuning models](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/finetune.md), [adding new dataset](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/new_dataset.md), [designing data pipeline](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/data_pipeline.md), [customizing models](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/customize_models.md), [customizing runtime settings](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/customize_runtime.md) and [useful tools](https://github.com/open-mmlab/mmdetection/blob/master/docs/useful_tools.md).



# 1. demo/MMDet_Tutorial.ipynb

- 































