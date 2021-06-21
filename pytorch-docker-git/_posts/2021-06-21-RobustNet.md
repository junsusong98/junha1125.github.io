---
layout: post
title: 【Pytorch】 RobustNet teardown reports

---

RobustNet teardown reports 

# RobustNet

## 1. docker build & run

```sh
$ sudo docker rename DA DA.Robust.Adapt.Adven
```



## 2. RobustNet github

코드 링크 : [https://github.com/shachoi/RobustNet](https://github.com/shachoi/RobustNet)

**Dataset Setting** - 해당 Readme.md [Link](https://github.com/shachoi/RobustNet#how-to-run-robustnet) 참조

Directory Tree

```shell
RobustNet
|-- split_data # gta5와 synthia 데이터셋에 대해서 어떤 이미지를 Train, Val로 사용할지 나눠줌
|   |-- gtav_split_test.txt
|   |-- gtav_split_train.txt
|   |-- gtav_split_val.txt
|   |-- synthia_split_train.txt
|   └-- synthia_split_val.txt
|-- datasets # 나중에 dataloder를 정의하기 위한, 각 Dataset의 API 코드들
|   |-- __init__.py  ## dataloder/transform 과 관련된 함수들 정의되어 있음
|   |-- bdd100k / camvid / cityscapes / cityscapes_labels
|   |-- gtav.py / kitti / mapillary / multi_loader / nullloader
|   └── sampler / synthia / uniform.py
|-- scripts # train.py를 위한 코드들
|   |-- infer_r50os16_cty_isw.sh # eval.py 호출
|   |-- train_mobile_gtav_base.sh # train.py 호출
|   |-- train_r101os8_cty_base.sh
|   |-- train_r50os16_cty_base.sh
|   |-- train_r50os16_gtav_base.sh
|   |-- train_r50os16_gtav_switchable.sh
|   └-- train_shuffle_gtav_base.sh
|-- transforms
|   |-- __init__.py
|   |-- joint_transforms.py
|   └-- transforms.py
|-- network
|   |-- Mobilenet.py
|   |-- Resnet.py
|   |-- SEresnext.py
|   |-- Shufflenet.py
|   |-- __init__.py
|   |-- cov_settings.py
|   |-- deepv3.py
|   |-- instance_whitening.py
|   |-- mynn.py
|   |-- switchwhiten.py
|   |-- sync_switchwhiten.py
|   └── wider_resnet.py
|-- config.py # train, eval.py 에서 assert_and_infer_cfg(args) 함수 호출. But 저자는 이거 사용 안 함. sh 사용함
|-- train.py
|-- eval.py
|-- loss.py
|-- optimizer.py
└-- utils
    |-- __init__.py
    |-- attr_dict.py
    |-- misc.py
    └-- my_data_parallel.py
```



**RobustNet/scripts/infer.sh**

```sh
# Testing for the GTA5-to-Cityscapes model (multi-level)
python -m torch.distributed.launch --nproc_per_node=1 eval.py \
    --dataset cityscapes \
    --arch network.deepv3.DeepR50V3PlusD \
    --inference_mode sliding \
    --scales 0.5,1.0,2.0 \
    --split val \
    --crop_size 1024 \
    --cv_split 0 \
    --ckpt_path ${2} \
    --snapshot ${1} \
    --wt_layer 0 0 2 2 2 0 0 \
    --dump_images \
```



**RobustNet/scripts/train.sh**

```sh
#!/usr/bin/env bash
    # Example on Cityscapes
     python -m torch.distributed.launch --nproc_per_node=2 train.py \
        --dataset cityscapes \
        --covstat_val_dataset cityscapes \
        --val_dataset bdd100k gtav synthia mapillary \
        --arch network.deepv3.DeepR50V3PlusD \
        --city_mode 'train' \
        --lr_schedule poly \
        --lr 0.01 \
        --poly_exp 0.9 \
        --max_cu_epoch 10000 \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 1024 \
        --crop_size 768 \
        --scale_min 0.5 \
        --scale_max 2.0 \
        --rrotate 0 \
        --max_iter 40000 \
        --bs_mult 4 \
        --gblur \
        --color_aug 0.5 \
        --wt_reg_weight 0.0 \
        --relax_denom 0.0 \
        --cov_stat_epoch 0 \
        --wt_layer 0 0 0 0 0 0 0 \
        --date 0101 \
        --exp r50os16_city_base \
        --ckpt ./logs/ \
        --tb_path ./logs/
```





## 3. Evaluate 흐름 따라가기

dd





## 4. Train 흐름 따라가기

dd





