---
layout: post
title: 【Pytorch】 RobustNt & ProDA & DA-SAC teardown reports
---

RobustNet teardown reports 

# 1. RobustNet

## 1.1. docker build & run

```sh
$ sudo docker rename DA "DA.Robust.Adapt.Adven"
```



## 1.2. RobustNet github

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
$ set -- <snapshot_path, saving_result_path> <ckpt_path>
$ python -m torch.distributed.launch --nproc_per_node=1 eval.py \
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
$ python -m torch.distributed.launch --nproc_per_node=2 train.py \
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





## 1.3. Evaluate 흐름 따라가기

\<전체 흐름 요약\>

1. `main` 
   - Network 정의 
   - dataloader(pbar) 정의
   - runner class 정의. inference에 필요한 맴버함수들 존재함
   - for dataloder: runner.inf(image)
   - inference(호출 함수가 저장된 변수)를 위한 방식으로 inference_sliding 함수를 사용한다.
   - 하나의 이미지를 Sliding window로 예측하고 결과를 융합하는 방식
2. `runner.inf`
   - inference(net, img, scales) 적용하여 한장의 이미지 Inference 결과 추출
3. `inference_sliding = inference`
   1. **for 이미지 Scale [0.5, 1, 2]:**
   2. ​     **for sliding_window_cropping:**
   3. ​            output_crop = model(input_crop)



\<나의 생각 정리\>

- 위와 같이 Inference를 하는데 옷갖 노력을 다하는데... 성능이 안나올 수가 있나??
  1. `Multi-Scale Testing`
  2. `Sliding window Predicting`
- "SOTA를 찍기위해서 정말 미친 노력을 하는구나" 라는 생각이 든다.
- 그럼에도 불구하고.. 모든 모델에 대해서 같은 프로세스를 적용하니, 그렇게 큰 문제는 아닌듯 하고. 모르겠다.
- 참고로 mmsegmentation [test.py](https://github.com/open-mmlab/mmsegmentation/blob/master/tools/test.py) 를 참고해보면 다음의 Inference 과정이 있는 것을 알 수 있었다. RobustNet은 Flip 까지는 적용하지 않았다.
  1. cfg.[aug-test](https://github.com/open-mmlab/mmsegmentation/blob/master/tools/test.py#L22): Use **Flip** and Multi scale aug
  2. cfg.[test_cfg](https://github.com/open-mmlab/mmsegmentation/blob/fb24bf54b69ad8e83098e4b18d789f25d0762874/docs/tutorials/config.md#an-example-of-pspnet): 어떤 inference 모드를 적용할지. (whole image or sliding windows) 





## 1.4. Train 흐름 따라가기

<**datasets/setup_loaders(args)**>

1. `arg.dataset` 에는 1.Cityscapes 2. Mapillary 3. gtav 4. synthia 5. bdd100k 6. null_loader 중 일부가 list에 담겨 있다. 이 list만, 새로운 list 변수에 dataloader를 모두 담아 놓는다.
2. `train dataloaders` 는 torch.utils.data.ConcatDataset 함수를 사용해서 한거뻔에 묶어버린다.
3. `val dataloaders`는 dictornary 자료형을 사용해서, 데이터셋 이름이 "Key", Dataloder를 "Value"로 저장해 놓는다.



**<train.py>**

1. `args.syncbn` 는 multi-gpu 사용 여부를 담은 변수로써 사용된다. True이면 Dataloader multi-gpu 사용을 위해, 별도의 sampler를 생성한다.
2. `main`
   - `data loader = datasets.setup_loaders(args)`  
   - `datasets.__init__.py `호출 



**<datasets.init.py>**

1. CityScape만해도 CityScapes / CityScapesUniform / CityScapesAug 이렇게 3가지 종류의 torch.utils.data.dataset 클래스가 있다. 
   1. CityScapes: Train Dataset 정의를 위한 단순한 Dataset
   2. CityScapesUniform: Class Imbalance 문제해결을 위해서, 한 Epoch에 들어가는 Class의 양이 비슷하게 1 Epoch용 데이터셋 새로 설정하는 데이터셋
   3. CityScapesAug: Test말고 Validation을 위해서 추가적인 Augmentation을 적용하는(?) 데이터셋
2. 다른 데이터셋도 똑같이 위와 같은 분류를 해놓는다. 



-> 일단 여기까지 보고, ProDA로 넘어가자





# 2. ProDA

Code Link: https://github.com/microsoft/ProDA

논문 설명 Post: [Prototypical Pseudo Label Denoising and Target Structure](https://junha1125.github.io/blog/artificial-intelligence/2021-04-02-ProDA/)

Directory Tree

```sh
ProDA
|-- calc_prototype.py
|-- data
|   |-- DataProvider.py
|   |-- __init__.py
|   |-- augmentations.py
|   |-- base_dataset.py
|   |-- cityscapes_dataset.py
|   |-- gta5_dataset.py
|   |-- randaugment.py
|   └-- synthia_dataset.py
|-- generate_pseudo_label.py
|-- models
|   |-- adaptation_modelv2.py
|   |-- deeplabv2.py
|   |-- discriminator.py
|   |-- sync_batchnorm
|   └-- utils.py
|-- metrics.py
|-- parser_train.py
|-- test.py
|-- train.py
└-- utils.py
```

학습 과정 및 추론 과정 

1. Pseudo label을 생성해야 하기 때문에, 몇가지 Step을 거쳐야 한다. 따라서 학습과정이 매우 복잡하다.
2. [학습과정 순서](https://github.com/microsoft/ProDA/blob/main/README.md#training)
3. [추론과정 순서](https://github.com/microsoft/ProDA/blob/main/README.md#inference-using-pretrained-model)

아래의 과정은 Target Structure 에서 **Origin Image & Augmentation Image pair를 어떻게 만드는지** 분석하기 위한 과정이다.



## 2.1 Target Structure 코드 분석 

1. [config file parser_train.py](https://github.com/microsoft/ProDA/blob/main/parser_train.py)

2. [ProDA/train.py](https://github.com/microsoft/ProDA/blob/main/train.py)    

   ```python
   datasets = create_dataset(opt, logger)
   for data_i in datasets.target_train_loader:
     	##### Dataloder ######################################################################
       # Source domain data는 아래의 "source_train_loader" 에서 가져온다.
       # Target domian data는 위의  "target_train_loader" 에서 가져온다.
       # 특히 target에서는 (1) target_image (2) target_imageS(Strong augmentation) 이 존재해서 Structure Learning을 가능케 한다.
   	    	### target Dataloder ###################
       target_image = data_i['img'].to(device)
       target_imageS = data_i['img_strong'].to(device)
       target_params = data_i['params']
       target_image_full = data_i['img_full'].to(device)
       target_weak_params = data_i['weak_params']
       target_lp = data_i['lp'].to(device) if 'lp' in data_i.keys() else None
       target_lpsoft = data_i['lpsoft'].to(device) if 'lpsoft' in data_i.keys() else None
       		### Source Dataloder ###################
       source_data = datasets.source_train_loader.next()
       model.iter += 1
       i = model.iter
       images = source_data['img'].to(device)
       labels = source_data['label'].to(device)
       source_imageS = source_data['img_strong'].to(device)
       source_params = source_data['params']
       
       ##### train ##########################################################################
       if opt.stage == 'warm_up':
           loss_GTA, loss_G, loss_D = model.step_adv(images, labels, target_image, source_imageS, source_params) # advent
       elif opt.stage == 'stage1':
           loss, loss_CTS, loss_consist = model.step(images, labels, target_image, target_imageS, target_params, target_lp,
                                                     target_lpsoft, target_image_full, target_weak_params)
       else:
           loss_GTA, loss = model.step_distillation(images, labels, target_image, target_imageS, target_params, target_lp)
   
       ##### Print Log ##########################################################################                
       ##### Evaluation #########################################################################
       validation(model, logger, datasets, device, running_metrics_val, iters = model.iter, opt=opt)
   ```

3. ProDA/models/adaptation_modelv2.py : 이전에 만든 Pseudo Label을 사용하고, Target Structure Learning (Contrastive Learning)을 수행한다.    

   ```python
   def step(self, source_x, source_label, target_x, target_imageS=None, target_params=None, target_lp=None, 
               target_lpsoft=None, target_image_full=None, target_weak_params=None):
       """
       Tip: 
       1. ema model이란 exponential moving averages of model parameters. 을 말한다
       2. opt.rce: symmetry cross entropy loss
       3. opt.S_pseudo: loss weight of pseudo label for strong augmentation
       Parameters:
       1. source_x : source domain 이미지
       2. source_label : source domain 이미지의 GT 
       3. target_x : Target domain 이미지
       4. target_imageS : Target domain 이미지 with Strong augmentation
       """
   ```

4. [ProDA/data/\_\_init\_\_.py](ProDA/data/__init__.py) 

   - create_dataset 정의되어 있음
   - self.target_train = find_dataset_using_name(opt.tgt_dataset)
   - "cityscapes_dataset.py" 에서 BaseDataset을 하위 클래스로 가지는 class인, Cityscapes_loader 호출

5. [ProDA/data/cityscapes_dataset.py](https://github.com/microsoft/ProDA/blob/main/data/cityscapes_dataset.py)      

   ```python
   class Cityscapes_loader(BaseDataset):
   	  def __getitem__(self, index):
           if self.augmentations!=None:
                       img, lbl, lp, lpsoft, weak_params = self.augmentations(img, lbl, lp, lpsoft)
                       img_strong, params = self.randaug(Image.fromarray(img)) # RandAugmentMC(2, 10)
                       img_strong, _, _ = self.transform(img_strong, lbl) 			# dataset 호출할 때, 추가하고 싶은 Augmentation
                       input_dict['img_strong'] = img_strong
   				input_dict['img'] = img
           input_dict['img_full'] = torch.from_numpy(img_full).float()
           input_dict['label'] = lbl_
           return input_dict
   ```








# 3. DA-SAC

논문 설명 Post: [Self-supervised Augmentation Consistency for DA](https://junha1125.github.io/blog/artificial-intelligence/2021-06-12-DA+MoCo/)

**Origin Image & Augmentation Image pair를 어떻게 만드는지** 분석하기 위한 과정이다.

[DA-SAC/datasets/dataloader_target.py](https://github.com/visinf/da-sac/blob/main/datasets/dataloader_target.py)

```python
## 아래의 3가지 Transform이 존재한다.
def __getitem__(self, index):
  self.tf_pre # 가장 기본적 augmentation
  self.tf_augm # clean image에 적용할 augmentation
  self.tf_post # Noisy image에 적용할 augmentaiton

  # 가장 기본 Augmentation 적용한 Iamge와 Label이 적혀 있는 Mask
  augms = self.tf_pre(images, masks) 
  
  affine_params = augms[-1]
  augms = augms[:-1]

  # 위 Image와 Mask 그대로
  augms2 = copy.deepcopy(augms) 
  # Strong augmentation 까지 적용한 Image와 Mask
  augms1 = self.tf_augm(*augms) 

  images1, masks = self.tf_post(*augms1)
  images2, _ = self.tf_post(*augms2)
  
  # Cleen Image로 부터 나온 Pseudo Label과 Strong augmentation image의 detection result를 매칭시켜줄 떄 필요한 듯 하다.
  affine = self._get_affine(affine_params)
  affine_inv = self._get_affine_inv(affine, affine_params)
  
  return images1, masks, images2, affine, affine_inv
```

















