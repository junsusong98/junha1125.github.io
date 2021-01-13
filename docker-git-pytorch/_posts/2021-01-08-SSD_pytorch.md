---
layout: post
title: 【Pytorch Package】SSD Pytorch Research / Detectron2 & mmdetection short research
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

- installation  
    ```
    $ cd SSD
    $ pip install -r requirements.txt
    $ python setup.py install  # or $ pip install .
    ```

# (Short) Detectron2 & mmdetection short research
- reference 
    - [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
    - [https://towardsdatascience.com/object-detection-in-6-steps-using-detectron2-705b92575578](https://towardsdatascience.com/object-detection-in-6-steps-using-detectron2-705b92575578)
    - [https://research.fb.com/wp-content/uploads/2019/12/4.-detectron2.pdf](https://research.fb.com/wp-content/uploads/2019/12/4.-detectron2.pdf)

1. Detectrion2 에서는 SSD, Yolo 와 같은 detector들은 제공하지 않는다. 
2. **it is important to try building a model at least once from scratch to understand the math behind it.**
3. [detectron2/MODEL_ZOO.md](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)에 보면 제공되는 모델과 그 성능을 확인할 수 있다. 
4. 이것을 이용해서 pretrain할 때, [cs231n-Transfer Learning](https://cs231n.github.io/transfer-learning/)을 참고하면 좋겠다. 나의 데이터량과 모델의 구조에 따라서 내가 어떤식으로 Transfer Learning을 진행해야하는지 나와있다.
5. 아래에 공부하는  lufficc/SSD 또한 detectron2와 거의 유사한 구조를 가지고 있다. 따라서 lufficc/SSD를 적극적으로 공부해도 되겠다 (어차피 detectron2에서 SSD를 제공하지 않으므로)
6. 하지만! mmdetection에서 SSD를 제공한다. 따라서 lufficc/SSD는 최대한 빠르게 넘어가고, mmdetection으로 넘어가는게 좋을 듯하다. mmdetection에서는 detection2보다 상대적으로 아주 많은 모델을 제공한다. 모델 비교는 아래의 링크에서 확인 가능 하다. 
    - [Models in mmdetection](https://github.com/open-mmlab/mmdetection#benchmark-and-model-zoo)
    - [Models in detectrion2](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)
7. detectron2와 mmdetection을 공부하기 전 논문을 읽어도 좋을 것 같다. 한글 블로그 정리 글도 많으니 참고하면 좋을 듯.
    - [detectron2 ppt](https://research.fb.com/wp-content/uploads/2019/12/4.-detectron2.pdf)
    - [mmdetection paper](https://arxiv.org/abs/1906.07155)
8. 따라서 앞으로 공부순서는 아래와 같다.
    1. lufficc/SSD
    2. mmdetection의 SSD
    3. detectrion의 fasterRCNN
9. detectron2/projects를 보면 \[DeepLab, DensePose, Panoptic-DeepLab, PointRend, TensorMask, TridentNet\] 와 같은 읽기만 해도 멋진 Detector or Segmentation 모델, 논문이 있으니 이것들을 공부해도 좋을 듯 하다. 
10. 특히 mmdetection에 대해서는 나중에 추가적으로 더 공부.




# 1. Analysis of lufficc/SSD
1. 독자적으로 사용하는 pypi 패키지가 2개 있다.
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
4.  ssd/data/datasets/coco.py & SSD/dataset/voc.py 
    - 각 데이터 셋을 사용하기 위한 함수들이 잘 정의되어 있다. 
    - Readme.md에 있는 [data directory 구조](https://github.com/lufficc/SSD#setting-up-datasets)를 똑같이 사용한다면, 나중에도 사용 가능! 
    - terminal 필수
        ```
        $ VOC_ROOT="/path/to/voc_root"
        $ export COCO_ROOT="/path/to/coco_root"
        ```

# 2. Package Github Exploration 
- file Tree  
    ```sh
    📦SSD
    ┣ 📂configs
    ┃ ┣ 📜efficient_net_b3_ssd300_voc0712.yaml
    ┃ ┣ 📜mobilenet_v2_ssd320_voc0712.yaml
    ┃ ┣ 📜mobilenet_v3_ssd320_voc0712.yaml
    ┃ ┣ 📜vgg_ssd300_coco_trainval35k.yaml
    ┃ ┣ 📜vgg_ssd300_voc0712.yaml
    ┃ ┣ 📜vgg_ssd512_coco_trainval35k.yaml
    ┃ ┗ 📜vgg_ssd512_voc0712.yaml
    ┣ 📂demo
    ┃ ┣ 📜000342.jpg
    ┃ ┣ 📜000542.jpg
    ┃ ┣ 📜003123.jpg
    ┃ ┣ 📜004101.jpg
    ┃ ┗ 📜008591.jpg
    ┣ 📂figures
    ┃ ┣ 📜004545.jpg
    ┃ ┣ 📜losses.png
    ┃ ┣ 📜lr.png
    ┃ ┗ 📜metrics.png
    ┣ 📂outputs
    ┃ ┗ 📜.gitignore
    ┣ 📂ssd
    ┃ ┣ 📂config
    ┃ ┃ ┣ 📜defaults.py
    ┃ ┃ ┣ 📜path_catlog.py
    ┃ ┃ ┗ 📜__init__.py
    ┃ ┣ 📂data
    ┃ ┃ ┣ 📂datasets
    ┃ ┃ ┃ ┣ 📂evaluation
    ┃ ┃ ┃ ┃ ┣ 📂coco
    ┃ ┃ ┃ ┃ ┃ ┗ 📜__init__.py
    ┃ ┃ ┃ ┃ ┣ 📂voc
    ┃ ┃ ┃ ┃ ┃ ┣ 📜eval_detection_voc.py
    ┃ ┃ ┃ ┃ ┃ ┗ 📜__init__.py
    ┃ ┃ ┃ ┃ ┗ 📜__init__.py
    ┃ ┃ ┃ ┣ 📜coco.py
    ┃ ┃ ┃ ┣ 📜voc.py
    ┃ ┃ ┃ ┗ 📜__init__.py
    ┃ ┃ ┣ 📂transforms
    ┃ ┃ ┃ ┣ 📜target_transform.py
    ┃ ┃ ┃ ┣ 📜transforms.py
    ┃ ┃ ┃ ┗ 📜__init__.py
    ┃ ┃ ┣ 📜build.py
    ┃ ┃ ┗ 📜__init__.py
    ┃ ┣ 📂engine
    ┃ ┃ ┣ 📜inference.py
    ┃ ┃ ┣ 📜trainer.py
    ┃ ┃ ┗ 📜__init__.py
    ┃ ┣ 📂layers
    ┃ ┃ ┣ 📜separable_conv.py
    ┃ ┃ ┗ 📜__init__.py
    ┃ ┣ 📂modeling
    ┃ ┃ ┣ 📂anchors
    ┃ ┃ ┃ ┣ 📜prior_box.py
    ┃ ┃ ┃ ┗ 📜__init__.py
    ┃ ┃ ┣ 📂backbone
    ┃ ┃ ┃ ┣ 📂efficient_net
    ┃ ┃ ┃ ┃ ┣ 📜efficient_net.py
    ┃ ┃ ┃ ┃ ┣ 📜utils.py
    ┃ ┃ ┃ ┃ ┗ 📜__init__.py
    ┃ ┃ ┃ ┣ 📜mobilenet.py
    ┃ ┃ ┃ ┣ 📜mobilenetv3.py
    ┃ ┃ ┃ ┣ 📜vgg.py
    ┃ ┃ ┃ ┗ 📜__init__.py
    ┃ ┃ ┣ 📂box_head
    ┃ ┃ ┃ ┣ 📜box_head.py
    ┃ ┃ ┃ ┣ 📜box_predictor.py
    ┃ ┃ ┃ ┣ 📜inference.py
    ┃ ┃ ┃ ┣ 📜loss.py
    ┃ ┃ ┃ ┗ 📜__init__.py
    ┃ ┃ ┣ 📂detector
    ┃ ┃ ┃ ┣ 📜ssd_detector.py
    ┃ ┃ ┃ ┗ 📜__init__.py
    ┃ ┃ ┣ 📜registry.py
    ┃ ┃ ┗ 📜__init__.py
    ┃ ┣ 📂solver
    ┃ ┃ ┣ 📜build.py
    ┃ ┃ ┣ 📜lr_scheduler.py
    ┃ ┃ ┗ 📜__init__.py
    ┃ ┣ 📂structures
    ┃ ┃ ┣ 📜container.py
    ┃ ┃ ┗ 📜__init__.py
    ┃ ┣ 📂utils
    ┃ ┃ ┣ 📜box_utils.py
    ┃ ┃ ┣ 📜checkpoint.py
    ┃ ┃ ┣ 📜dist_util.py
    ┃ ┃ ┣ 📜logger.py
    ┃ ┃ ┣ 📜metric_logger.py
    ┃ ┃ ┣ 📜misc.py
    ┃ ┃ ┣ 📜model_zoo.py
    ┃ ┃ ┣ 📜nms.py
    ┃ ┃ ┣ 📜registry.py
    ┃ ┃ ┗ 📜__init__.py
    ┃ ┗ 📜__init__.py
    ┣ 📜demo.py
    ┣ 📜DEVELOP_GUIDE.md
    ┣ 📜requirements.txt
    ┣ 📜setup.py
    ┣ 📜test.py
    ┣ 📜train.py
    ```  

1. ssd/modeling/detector
    - **ssd/modeling**에는 아래와 같은 신경망 구성 요소를 nn.module로 구현해놓은 파일이 있다. 
        1. anchors
        2. backbone
        3. box_head
    - ssd_detector.py 에서 nn모듈"SSDDetector"을 return 해줌.
        - SSDDetector는 아주 짧은 nn모듈이며, 아래의 2 모듈을 사용
            - from ssd.modeling.backbone import build_backbone
            - from ssd.modeling.box_head import build_box_head 
    - 따라서 신경망 구현에 대한 정확한 코드를 보고 싶다면, 위의 3개의 폴더 내부의 파일들을 봐야한다.
    - 이 폴더 내부 파일에 대해서는 아래의 큰 chapter로 다시 다룰 예정

2. ssd/utils   
    - checkpoint.py import CheckPointer : ckpt의 링크에서 모델 paramerter를 다운받고, 나의 신경망에 넣는 함수
        - model_zoo 파일로 가서, pth파일을 download하고 받은 cached_file을 return받는다. 
        - torch.load 함수를 사용한다. 
    - model_zoo.py
        - torch.hub 모듈을 사용한다. 
        - 이 모듈에는 download_url_to_file/ urlparse/ HASH_REGEX 와 같은 함수가 있다.
        - 나의 신경망 파라미터를 pht파일로 저장하고, 그것을 github에 올려놓고 누군가가 나의 신경망 파라미터를 사용할 수 있게 하려면, 이 torch.hub모듈을 사용해야겠다. 
    - registry.py
        - 아래 \[3. Analysis of lufficc/SSD/ssd/modeling\]에 있는 설명 참조

3. ssd/data/transforms
    - transforms.py 
        -torchvision.transfoms 에 있을 법한 함수들이 직접 만들어져 있다. 
        - Ex) resize, ToTensor, RandomFlip.. 과 같은 **클래스**들이 직접 구현되어 있다. 
        - 특히 compose 또한 직접 구현해놓았다. 
        - 위의 클래스들은 모두 \_\_call\_\_(parameters) 들이 구현되어 있다. def과 다르게 class로 만들어 놓은 클래스는 call을 정의해야만, 함수 포인터로써 클래스를 사용할 수 있다. 예를 들어 아래와 같이 사용이 가능하다.   
            ```python
            class ToTensor(object):
                def __call__(self, cvimage, boxes=None, labels=None):
                    return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels

            transform = [RandomSampleCrop(),
                            RandomMirror(),
                            ToTensor()]
            
            transform = Compose(transform)
            return transform
            ```
    - \_\_init\_\_.py : 
        - build_transforms, build_target_transform 와 같은 함수들이 정의되어 있고, 다른 파일에서 이 함수만 사용함으로써 쉽게 transform을 수행할 수 있다. 

4.


# 3. Analysis of lufficc/SSD/demo.py
- ```
    $ python demo.py \
        --config-file configs/vgg_ssd300_voc0712.yaml \
        --images_dir demo \
        --ckpt https://github.com/lufficc/SSD/releases/download/1.2/vgg_ssd300_voc0712.pth
    $ python demo.py \
        --config-file configs/vgg_ssd512_voc0712.yaml \
        --images_dir demo \
        --ckpt https://github.com/lufficc/SSD/releases/download/1.2/vgg_ssd512_voc0712.pth
    ```
- def main()
    1. [argparse 나의 포스트](https://junha1125.github.io/artificial-intelligence/2020-05-14-argparse/) 
    2. load config file that include info like num_classes, Dataset... 
    3. print   
        ```
        Loaded configuration file configs/vgg_ssd300_voc0712.yaml

        MODEL:
        NUM_CLASSES: 21
        INPUT:
        IMAGE_SIZE: 300
        ...
        ```
- def run_demo(cfg, ckpt, score_threshold, images_dir, output_dir, dataset_type)  
    1. @torch.no_grad() : **decorator**, Because it **define \_\_enter\_\_, \_\_exit\_\_** on [code here](https://pytorch.org/docs/stable/_modules/torch/autograd/grad_mode.html#no_grad).
    2. model = build_detection_model(cfg)
        - modeling/detector/\_\_inti\_\_.py에서 modeling/deector/ssd_detector.py의 nn모듈"SSDDetector"을 return 해줌
        - 이 모듈의 전체 신경망 코드는 나중에 공부.
    3. ssd.utils.checkpoint import CheckPointer 로 신경망 파라메터 load.
    4. transforms = build_transforms(-> compose([transforms]))
    5. [glob.glob(directory_path)](https://itholic.github.io/python-listdir-glob/) == os.listdir(directory path) 
    6. for i, image_path in enumerate(image_paths : list of images_path):
        - os.path.basename : 마지막 파일명만 return ('Module' Research 참고)
        - result = model(images.to(device))[0]
        - boxes, labels, scores = result['boxes'], result['labels'], result['scores']
        - drawn_image = draw_boxes(image, boxes, ...)
        - Image.fromarray(drawn_image).save(path)

# 3. Analysis of lufficc/SSD/ssd/modeling
- registry - 모듈을 config의 dictionary구조처럼 저장해 놓고, 쉽게 불러와 사용할 수 있게 해놓은 툴. 
    - 이와 같이 사용함  
        ```python
        # ssd/modeling/backbone/vvg.py
        @registry.BACKBONES.register('vgg')
        def vgg(cfg, pretrained=True):
            model = VGG(cfg)  # 같은 파일에서 정의한 클래스
            if pretrained:
                model.init_from_pretrain(load_state_dict_from_url(model_urls['vgg']))
            return model

        # ssd/modeling/__init__.py
        def build_backbone(cfg):
            return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
        ```
    - registry에 모듈을 저장해두고, config에 적혀있는데로, 각각의 상황마다 각각의 모듈을 호출하기 쉽게 만들어 놓음. switch문이나 if문을 여러개써서 어떤 boakbone을 string으로 입력했는지 확인하는 작업이 필요없다. 
    - 어려울 것 없고, 이 registry도 하나의 dictionary이다. 전체 코드에서는 dict{dict, dict, dict, dict ...} 와 같은 구조로 사용 중.
- 




















# 4. Analysis of lufficc/SSD/train.py
# 5. Analysis of lufficc/SSD/inferene.py