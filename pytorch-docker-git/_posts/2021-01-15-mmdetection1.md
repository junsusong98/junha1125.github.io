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



# 0. colab 전용 환경 설정

아래에 코드에 간단한, 에러해결의 고민이 적혀있다. 참고해도 좋지만, 새로운 환경에서 mmcv와 mmdetection을 설치하기 위해서, 그냥 주어진 mmcv와 mmdetection의 \[github, official_document\] 자료를 다시 읽어보고 공부해보는게 더 좋을 듯하다. 

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/GCPcode/torch_package
!ls

# !pip install torch==1.6.0 torchvision==0.7.0 # -> 오류 발생 이거 때문인지는 모름.
!pip install -U torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
import torch ; torch.__version__
# Colab : 아래 나오는 Restart runtime 눌러야 버전 변경 됨.

import torch ; import torchvision
torch.__version__, torchvision.__version__

# Check nvcc version
!nvcc -V
# Check GCC version
!gcc --version

# colab error defence
!pip install addict==2.2.1
!pip install yapf==0.28.0
!pip install Pillow==7.0.0
from addict import Dict
from yapf.yapflib.yapf_api import FormatCode

"""
# 이렇게 설치해서 그런지 에러가 나는데..? GCC, GUDA 버전문제라고? torch, GUDA 버전문제라고?
#!git clone https://github.com/open-mmlab/mmcv.git
%cd /content/drive/MyDrive/GCPcode/torch_package/mmcv
!MMCV_WITH_OPS=1 python setup.py develop
# Finished processing dependencies for mmcv-full==1.2.5
# 에러 이름 : linux-gnu.so: undefined symbol 
# 해결 : mmcv.__version : 1.2.5 말고 1.2.6으로 설치하게 두니 설치 완료.
# git으로 설치해서 직접 빌드하는 거는 왜... 1.2.5로 다운받아지는 거지? 모르겠다.
"""
# mmcv Readmd.md 처럼 위의 셀로, 쿠타, 토치 버전을 알고 mmcv를 정확하게 설치하는 것도 좋은 방법이다.
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
!pip install mmcv-full # colab-tuto에는 그냥 이렇게만 설치하니까 나도 일단 이렇게 설치.
# mmcv.__version__ : 1.2.6

import mmcv
mmcv.__version__

%cd /content/drive/MyDrive/GCPcode/torch_package
#!git clone https://github.com/open-mmlab/mmdetection.git
%cd /content/drive/MyDrive/GCPcode/torch_package/mmdetection

!pip install -r requirements/build.txt
!pip install -v -e .  # or "python setup.py develop"

import mmdet 
mmdet.__version__
# mmdet 2.8.0

!python mmdet/utils/collect_env.py # truble shooting

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

# Check installation
# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
# Check MMDetection installation
import mmdet
print(mmdet.__version__)
# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())
```





## 0.0 **colab코렙에서 간단한 debugging디버깅 방법** 

- ```python
  def func1(a, b):
      return a / b
  
  def func2(x):
      a = x
      b = x - 1
      return func1(a, b)
  
  #########
  
  func2(1)
  
  #########
  
  %debug
  ```

- 마지막 셀에 아래를 치면 바로 위에서 일어났던 에러 직전의 변수들을 직접 검색해서 찾아볼 수 있다.

- ```python
  %xmode Plain
  %pdb on
  func2(1)
  ```

- 이와 같이 먼저 위에 실행해 놓으면, 에러가 발생한 후 바로 interact모드로 들어간다.

- interact모드에서 사용할 수 있는 약어는 다음과 같은 것들이 있다.        

  ![image-20210128202827731](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210128202827731.png)

- 혹은 파이썬 코드에 import pdb; pdb.set_trace() 을 넣어둠으로써, break point를 설정할 수 있다. (근데 이렇게까지 할 거면, editor를 어떻게든 이용하는 편이 낫겠다. docker든 ssh든 다양한 방법이 있으니까)





# 1. demo/MMDet_Tutorial.ipynb

## 1.1 only inference using pretrained_model 

```python
%cd /content/drive/MyDrive/GCPcode/torch_package/mmdetection

!mkdir checkpoints
!wget -c http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth \
      -O checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

Choose to use a config and initialize the detector

config = 'configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py'

checkpoint = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

model = init_detector(config, checkpoint, device='cuda:0')

img = 'demo/demo.jpg'

result = inference_detector(model, img)

show_result_pyplot(model, img, result, score_thr=0.3)

# ** What is 'result'? **
"""
type(result) -> tuple
len(result)  -> 2
type(result[0]), type(result[1])  -> (list, list)
len(result[0]), len(result[1]) -> (80,80)
print(result[0]) # [confidence 값, bountding_box_position]
print(result[1]) # 80개 객체에 대한 w * h * channel(80) 의 bool type의 mask 정보
"""
```

- 원하는 모델 Inference 하는 마법같은 방법 ([mmdetection/mmddet/apis/inference](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/apis/inference.py))
  1. 핵심 모듈 2개만 import한다. `inference_detector, init_detector, show_result_pyplot`
  2. config파일은 [mmdetection/config](https://github.com/open-mmlab/mmdetection/tree/master/configs) 에서 골라서 가져오기, pth파일은 미리 다운받아 놓기.
  3. **init_detector**으로 model 생성하기
  4. **inference_detector**으로 원하는 이미지 추론
  5. **show_result_pyplot**으로 result 시각화



## 1.2 Train a detector on customized dataset

1. Modify the config

   - mmdetection에서는 config 파일을 .py 파일을 사용한다. 

   - 이 파일은 꼭 `from mmcv.Config import fromfile`파일과 함께 사용된다. 

   - ```python
     from mmcv import Config
     cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')
     ```

   - 이 다음에 cfg 파일을 수정하고 싶으면, **. (dot)** 을 이용해서 수정이 가능하다. 

   - ```python
     from mmdet.apis import set_random_seed
     
     # Modify dataset type and path
     cfg.dataset_type = 'KittiTinyDataset'
     cfg.data_root = 'kitti_tiny/'
     ```

   - 



























