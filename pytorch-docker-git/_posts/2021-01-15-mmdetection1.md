---

layout: post
title: 【mmdetection】mmdetection Tutorial and Overview
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
4. **<u>느낀점</u>** ⭐⭐
   - 생각보다, 거의 Torch, Torchvision 와 같은 **큰~모듈** 처럼, 내부를 모르는 채로 원하는 함수를 구글링, Official Document 검색으로 찾아서 해야할 만큼 큰 모듈이다. 내가 원하는 모델을 분석하기 위해서 이 패키지를 사용한다는 것은... 멍청한 행동인 것 같다. 그만큼 사소한거 하나하나가 모두 구현되어 있는 **큰~모듈**이다. 
   - 만약 내가 어떤 신경망 모델의 내부 코드가 궁금하다면, Github에서 그 신경망 모델의 이름을 검색하고,  그 신경망만을 위한 코드를 보는것이 낫겠다. 
   - 이전에는 이런 생각을 했다. "**만약 SSD를 공부하고 싶다면, SSD 패키지를 Github에서 검색해서 사용하는 것 보다는, mmdetection에서 SSD가 어떻게 구현되어 있는지, 어떤 모듈을 사용하는지 찾아보는게 더 좋지 않나? 그게 더 안정화된 코드고 빠른 코드 아닐까? 그래야 내가 혹시 필요한 모듈을 그대로 가져와서 쓸 수 있지 않을까??**" 라고 생각했다. **물론 맞는 말이다....** 
   - 하지만 지금 시간이 급하니.. 정말 필요할 때, '2. docs/with existing dataset.md' 에서 나온 방법을 이용해서, test.py와 train.py를 디버깅하고 어떤 흐름으로, 어떤 함수와 클래스를 이용하며 학습이 되고, 테스트가 되는지 찾아보는 시간을 가지는 것도 좋을 듯 하다.
   - 그럼에도 불구하고, **만약 내가 BlendMask 라는 함수를 수정해서 , 선배님처럼 The Devil Boundary Mask 모델을 만들고 싶다면 mmdetection이나, detectron2를 사용하지 않아도 될 수 있다. 하지만 그래도!! Developer for practice로서, mmdetection과 detectrion2 사용법과 코드가 돌아가는 내부 흐름은 알아두어야 한다고 생각한다.** 
   - 따라서 개인 컴퓨터가 생기면 디버깅을 하면서, 직접 내부 흐름을 살펴보는 시간도 가져보자.



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

1. **Modify cfg**

   - mmdetection에서는 config 파일을 .py 파일을 사용한다. 

     - 이 파일은 꼭 `from mmcv.Config import fromfile`파일과 함께 사용된다. 

     - mmcv.Config.fromfile path  -> 그냥 쉽게 생각하며 `dictionary`이다!

     - [fast_rcnn_r50_caffe_fpn_1x_coco.py](https://github.com/open-mmlab/mmdetection/blob/master/configs/fast_rcnn/fast_rcnn_r50_caffe_fpn_1x_coco.py) 기에 들어가봐도, dict이라는 dictionary생성자를 이용해서 config파일을 생성한다. 

     - ```python
       from mmcv import Config
       cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')
       ```

     - cfg.data 를 보면 아래와 같은 Key, Value가 있는 것을 확인할 수 있다. 

      - ```python
        type(cfg) # cv.utils.config.Config
        type(cfg.data) # mmcv.utils.config.ConfigDict
        cfg.data 
        """
        {'samples_per_gpu': 2,
         'test': {'ann_file': 'data/coco/annotations/instances_val2017.json',
          'img_prefix': 'data/coco/val2017/',
          'pipeline': [{'type': 'LoadImageFromFile'},
           {'flip': False,
            'img_scale': (1333, 800),
            'transforms': [{'keep_ratio': True, 'type': 'Resize'},
             {'type': 'RandomFlip'},
             {'mean': [103.53, 116.28, 123.675],
              'std': [1.0, 1.0, 1.0],
              'to_rgb': False,
              'type': 'Normalize'},
             {'size_divisor': 32, 'type': 'Pad'},
             {'keys': ['img'], 'type': 'ImageToTensor'},
             {'keys': ['img'], 'type': 'Collect'}],
            'type': 'MultiScaleFlipAug'}],
          'type': 'CocoDataset'},
         ....
         ....
        ```
        
     - 위에서 확인한 Key를 아래와 같이 수정할 수 있다. **. (dot)** 을 이용해서 수정이 가능하다. 

     - ```python
          cfg.dataset_type = 'KittiTinyDataset'
          cfg.data_root = 'kitti_tiny/'
          
          cfg.data.test.type = 'KittiTinyDataset'
          cfg.data.test.data_root = 'kitti_tiny/'
          cfg.data.test.ann_file = 'train.txt'
          cfg.data.test.img_prefix = 'training/image_2'
          
          ... (mmdetection/demo/MMDet_Tutorial.ipynb 파일 참조)
          
          # The original learning rate (LR) is set for 8-GPU training.
          # We divide it by 8 since we only use one GPU.
          cfg.optimizer.lr = 0.02 / 8
          cfg.lr_config.warmup = None
          cfg.log_config.interval = 10
          
          # Change the evaluation metric since we use customized dataset.
          cfg.evaluation.metric = 'mAP'
          # We can set the evaluation interval to reduce the evaluation times
          cfg.evaluation.interval = 12
          # We can set the checkpoint saving interval to reduce the storage cost
          cfg.checkpoint_config.interval = 12
          
          # print(cfg)를 이쁘게 보는 방법
          print(f'Config:\n{cfg.pretty_text}')
       ```

     - 여기서 핵심은, <u>cfg.data.test/train/val.type = '내가 아래에 만들 새로운 dataset'</u> 을 집어 넣는 것이다.

2. **Regist Out Dataset**

   - 데이터 셋을 다운로드 한후, 우리는 데이터 셋을  COCO format, middle format으로 바꿔줘야 한다. 

   - 여기서는 아래이 과정을 수행한다.

     1. **from** mmdet.datasets.custom **import** CustomDataset
     2. CustomDataset 을 상속하는 클래스를 만들고 `def load_annotations` 해주기

   - middle format MMDetection 은 아래와 같은 format이다.  [1. cocodata format arrangement](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch), [2. coco format official](https://cocodataset.org/#format-data)과 비슷하다.

     - ```yaml
       [
           {
               'filename': 'a.jpg',
               'width': 1280,
               'height': 720,
               'ann': {
                   'bboxes': <np.ndarray> (n, 4),
                   'labels': <np.ndarray> (n, ),
                   'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                   'labels_ignore': <np.ndarray> (k, 4) (optional field)
               }
           },
           ...
       ]
       ```

   - load_annotations를 정의하는 과정과 코드는 아래와 같다

     - 코드 요약 : 

       1. 아래의 load_annotations에 들어가는 매개변수로, ann_file(type : str)에는 여기서 지정한 파일이 들어간다. `cfg.data.test.ann_file = 'train.txt'`
       2. train.txt파일의 첫줄을 살펴보면 아래와 같다. `Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01`
       3. data_infos는 dict들을 저장해놓은 list이다. In detail, **data_infos에는 \[ <u>1장</u>에 이미지에 있는 <u>객체들</u>의 정보를 저장해둔, data_info=dictionary\]들이 원소 하나하나로 들어간다.** 
       4. data_info의 key는 filename, width, height, ann 이다. 특히 ann의 key는 bboxes, labels 등이 있다.
       5. (`self.img_prefix` 와 같은 CustomDataset 클래스의 맴버변수가 쓰여서 디버깅 못함) 

     - 코드 :

       - ```python
         import copy
         import os.path as osp
         
         import mmcv
         import numpy as np
         
         from mmdet.datasets.builder import DATASETS
         from mmdet.datasets.custom import CustomDataset
         
         @DATASETS.register_module()
         class KittiTinyDataset(CustomDataset):
         
             CLASSES = ('Car', 'Pedestrian', 'Cyclist')
         
             def load_annotations(self, ann_file):
                 cat2label = {k: i for i, k in enumerate(self.CLASSES)}
                 # load image list from file
                 image_list = mmcv.list_from_file(self.ann_file)
             
                 data_infos = []
                 # convert annotations to middle format
                 for image_id in image_list:
                     filename = f'{self.img_prefix}/{image_id}.jpeg'
                     image = mmcv.imread(filename)
                     height, width = image.shape[:2]
             
                     data_info = dict(filename=f'{image_id}.jpeg', width=width, height=height)
             
                     # load annotations
                     label_prefix = self.img_prefix.replace('image_2', 'label_2')
                     lines = mmcv.list_from_file(osp.join(label_prefix, f'{image_id}.txt'))
             
                     content = [line.strip().split(' ') for line in lines]
                     bbox_names = [x[0] for x in content]
                     bboxes = [[float(info) for info in x[4:8]] for x in content]
             
                     # 진짜 필요한 변수
                 	gt_bboxes = []
                     gt_labels = []
                     # 사용하지 않은 변수
                     gt_bboxes_ignore = []
                     gt_labels_ignore = []
             
                     # filter 'DontCare'
                     for bbox_name, bbox in zip(bbox_names, bboxes):
                         if bbox_name in cat2label:
                             gt_labels.append(cat2label[bbox_name])
                             gt_bboxes.append(bbox)
                         else:
                             gt_labels_ignore.append(-1)
                             gt_bboxes_ignore.append(bbox)
         
                     data_anno = dict(
                         bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                         labels=np.array(gt_labels, dtype=np.long),
                         bboxes_ignore=np.array(gt_bboxes_ignore,
                                                dtype=np.float32).reshape(-1, 4),
                         labels_ignore=np.array(gt_labels_ignore, dtype=np.long))
         
                     data_info.update(ann=data_anno)
                     data_infos.append(data_info)
         
                 return data_infos
         ```

3.  **Train Our Model**

   - ```python
     from mmdet.datasets import build_dataset
     from mmdet.models import build_detector
     from mmdet.apis import train_detector
     ```

   - 위의 핵심 모듈을 사용해서, 우리가 정의한 dataset을 학습시켜보자

   - ```python
     # Build dataset
     datasets = [build_dataset(cfg.data.train)]
     
     # Build the detector
     model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
     # Add an attribute for visualization convenience
     model.CLASSES = datasets[0].CLASSES
     
     # Create work_dir
     mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
     train_detector(model, datasets, cfg, distributed=False, validate=True)
     ```

   - 실행하면, 우리가 설정한 epoch=12로 12번 학습된다. 

4. **Inference with Out Model**

   - ```python
     img = mmcv.imread('kitti_tiny/training/image_2/000068.jpeg')
     
     model.cfg = cfg
     result = inference_detector(model, img)
     show_result_pyplot(model, img, result)
     ```





# 2. docs/with existing dataset.md

1. 위와 똑같은 Just Inference 하는 방법
2. Asynchronous interface 
   - Inference를 동시에, 다른 input으로, 혹은 다른 model을 사용하여 돌릴 수 있는 방법이다,
   -  `import asyncio` 라는 모듈을 사용해서 thread와 CPU, GPU의 적절한 사용을 이뤄낸다. 
   - (아직 내가 사용할 방법은 아닌 것 같음)
3. Test existing models on standard datasets 
   - public datasets including COCO, Pascal VOC, CityScapes, and [more dataset 종류](https://github.com/open-mmlab/mmdetection/tree/master/configs/_base_/datasets) 를 사용하는 방법. 그리고 이 dataset을 [어떤 Path에다가 위치시켜야 하는지](https://github.com/open-mmlab/mmdetection/blob/master/docs/1_exist_data_model.md#prepare-datasets).
   - `tools/test.py` 등을 이용해 test를 진행한다. 
   - 다양한 상황에 대해서, argparse를 어떻게 집어넣어줘야 하는지 있으니 참고
4. Train predefined models on standard datasets
   - `tools/train.py` 등을 이용해 train를 진행한다. 많은 예시들이 추가되어 있으니 참고
   - 다양한 상황에 대해서, argparse를 어떻게 집어넣어줘야 하는지 있으니 참고



# 3. Train with customized datasets

1. detectron2에서 봤던것 처럼, [ballon dataset](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon) 을 사용해서 공부한다,

2. STEP

   1. Prepare the customized dataset
   2. Prepare a config
   3. Train, test, inference models on the customized dataset.

3. (1) Prepare the customized dataset

   - coco format으로 ([1. cocodata format arrangement](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch), [2. coco format official](https://cocodataset.org/#format-data)) data format을 바꿔줘야 한다. 
   - coco_json_file 에 image의 path와 image의 annotation정보가 모두 있기 때문에, 사실 우리의 python 코드는 json 하나만 보면 된다!!
   - 따라서 CSV이든 XML이든 어떤 파일 형식으로 annotation된 데이터가 있다면, coco 즉 json포멧으로 annotation을 바꾸는 코드가 분명 있을거야. github에도 있을거고, 나의 블로그 Post에 【Tensorflow】, 【Keras】파일에도 이에 대한 내용이 잇었으니까 참고하도록 하자. 
   - mmdetection에서도 mmcv를 이용해서 어떤 파일형식이든, coco format형식으로 바꿔주는, `def convert_balloon_to_coco(ann_file, out_file, image_prefix):` 함수를 예시로 주어주었다. 가능하다면 이것을 사용해도 좋은 듯 하다.

4. (2) Prepare a config

   - ballon dataset을 사용하기 위해, 어떻게 `mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon.py` 파일을 만들었는지 나와 있으니 참고하도록 하자. ([config.py 링크](https://github.com/open-mmlab/mmdetection/blob/master/docs/2_new_data_model.md#prepare-a-config))
   - 위의 KittiTiny는 `from mmdet.datasets.custom import CustomDataset` 를 상속받아서 dataset을 만들어서 그런지, the number of class에 대한 고려는 안했지만, 여기서는 json파일을 직접 만들었기 때문에 #class를 고려를 해줬다. 

5. (3) Train, test, inference models on the customized dataset.

   - ```sh
     $ python tools/train.py configs/ballon/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon.py
     $ python tools/test.py configs/ballon/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon.py work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon.py/latest.pth --eval bbox segm
     ```

   - 더 다양한 case에 대한 agupare 적용방법은, 2. docs/with existing dataset.md 파일 참조





# 4. **webcam_demo.py** 

- ```python
   model = init_detector(args.config, args.checkpoint, device=device)
    
  camera = cv2.VideoCapture(args.camera_id)
  
  print('Press "Esc", "q" or "Q" to exit.')
  while True:
      ret_val, img = camera.read()
      result = inference_detector(model, img)
      ch = cv2.waitKey(1)
      if ch == 27 or ch == ord('q') or ch == ord('Q'):
          break
      model.show_result( img, result, score_thr=args.score_thr, wait_time=1, show=True)
  ```

- 매우 간단.





# 5. mmdetection for SSD

- [2021-01-29-SSDwithCode](https://junha1125.github.io/blog/artificial-intelligence/2021-01-29-SSDwithCode/#2-mmdetection-for-ssd)























