---
layout: post
title: 【Pytorch】 AdaptSeg & ADVENT teardown reports

---

AdaptSeg teardown reports 

ADVENT teardown reports

# 1. AdaptSeg

## 1-1. docker build & run

```sh
$ sudo docker run -d -it\
--gpus all\
--restart always\
-p 8880:8080\
--name DA\
--shm-size 8gb\
-v /home/junha/docker:/workspace\
-v /hdd1T:/dataset\
pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel
```





## 1-2. AdaptSeg github

코드 링크 : [https://github.com/wasidennis/AdaptSegNet](https://github.com/wasidennis/AdaptSegNet)

**Dataset Setting**

```sh
$ mkdir data
$ ln -s /dataset/sementic/data/ ./data
```

**Evaluate AdaptSeg**

```sh
# Testing for the GTA5-to-Cityscapes model (multi-level)
$1. python evaluate_cityscapes.py --restore-from /workspace/AdaptSegNet/checkpoints/GTA2Cityscapes_oracle/GTA5_95000.pth
# Compute the IOU
$2. python compute_iou.py ./data/Cityscapes/gtFine/val result/cityscapes
$2. python compute_iou.py /workspace/data/Cityscapes/gtFine/val result/cityscapes
```

**Train AdaptSeg**

```sh
python train_gta2cityscapes_multi.py --snapshot-dir ./checkpoints/GTA2Cityscapes_multi \
                                     --lambda-seg 0.1 \
                                     --lambda-adv-target1 0.0002 \
                                     --lambda-adv-target2 0.001\
                                     --gan LS\
                                     --tensorboard\
```





## 1-3. Evaluate 흐름 따라가기

**3.1 evaluate_cityscapes.py**

- 아래의 3개의 모델 중 하나를 사용한다. 3개는 모두 backbone만 다를 뿐, 별 차이 없다.
  1. `model/deeplab.py`
  2. `model/deeplab_multi.py`
  3. `model/deeplab_vgg.pu`
- Dataloader = cityscapes 을 위해서만 정의한다. 
- Dataset 정의는 `dataset/cityscapes_dataset.py` 에서 이뤄지는데. 나중에 필요하면! 공부.
- 너무 심플하고 단순한 코드여서, 할말이 없다. 쩝.



**3.2 model/deeplab_multi.py**

- Backbone: Resnet 101 
- Classifier_Module == ASPP_Module == Pixel Level classification moduels



**3.3 compute_iou.py**

- (각 pixel에 class 번호가 적힌) 1024 * 2048 * 1 이미지를 비교해서 miou를 계산해 출력해준다. (계산하는 원리는 굳이 안봄. 필요하면 찾아보자.)

- Predicted name: `result/cityscapes/frankfurt_000001_007973_leftImg8bit.png?raw=tru`

- GT image name: `./data/Cityscapes/gtFine/val/frankfurt/frankfurt_000001_007973_gtFine_labelIds.png?raw=tru`

- 출력결과 예시   

  ```txt
  ===>road:       86.46
  ===>sidewalk:   35.96
  ===>building:   79.92
  ===>wall:       23.41
  ===>fence:      23.27
  ===>pole:       23.87
  ===>light:      35.24
  ===>sign:       14.77
  ===>vegetation: 83.35
  ===>terrain:    33.25
  ===>sky:        75.62
  ===>person:     58.49
  ===>rider:      27.55
  ===>car:        73.65
  ===>truck:      32.48
  ===>bus:        35.42
  ===>train:      3.85
  ===>motocycle:  30.05
  ===>bicycle:    28.11
  ===> mIoU: 42.35
  ```

  



## 1-4. Train 흐름 따라가기

**4.1 train_gta2cityscapes_multi.py**

- 전체 과정 요약 아래 참조, 아래 ADVENT 그림이 바로 아래 사진에서 좀 보정한 것. (1번 ~ 10번까지 순서대로 따라가기)
  ![image-2021061214485613](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora-rcv/image-2021061214485613.jpg?raw=true)



**Training AdaptSeg**

```sh
python train_gta2cityscapes_multi_AdaptSeg.py --snapshot-dir ./checkpoints/GTA2Cityscapes_adaptseg \
                                     --lambda-seg 0.1 \
                                     --lambda-adv-target1 0.0002 \
                                     --lambda-adv-target2 0.001\
                                     --gan LS\
                                     --tensorboard\
                                     --log-dir ./log/adaptseg
```

**Training SourceOnly**

```sh
python train_gta2cityscapes_multi_SourceOnly.py --snapshot-dir ./checkpoints/GTA2Cityscapes_source \
                                     --lambda-seg 0.001 \
                                     --lambda-adv-target1 0.0002 \
                                     --lambda-adv-target2 0.001\
                                     --gan LS\
                                     --tensorboard\
                                     --log-dir ./log/source
```

**Training Oracle (Using Target)**

```sh
python train_gta2cityscapes_multi_Oracle.py --snapshot-dir ./checkpoints/GTA2Cityscapes_oracle \
                                     --lambda-seg 0.1 \
                                     --lambda-adv-target1 0.0002 \
                                     --lambda-adv-target2 0.001\
                                     --gan LS\
                                     --tensorboard\
                                     --log-dir ./log/oracle
```





# 2. ADVENT 

- 이 코드의 base는 AdeptSeg 이다. 

- 전체 구조 파악하기          

  ```sh
  ADVENT
  ├── dataset : cityscapes_list / gta5_list / torch.utils.data.dataset 정의
  └── domain_adaptation : config.py / eval_UDA.py / train_UDA.py
  └── model
  │		└── deeplabv2.py
  │		└── discriminator.py
  └── scripts : test.py / train.py 여기서 eval/train_UDA.py 코드 함수가 호출된다.
  └── utils : loss.py 및 다른 작은 일을 하는 함수들 모음
  ```

- 실행 코드    

  ```sh
  $ python train.py --cfg ./configs/advent.yml --tensorboard
  $ python test.py --cfg ./configs/advent.yml
  ```



## 2.1 train 흐름 따라가기

**아래 그림에서 빨간색 숫자 \[1 -> 10\] 따라가며 익히기**

![image-20210620134244562](/Users/junha/Library/Application Support/typora-user-images/image-20210620134244562.png)









