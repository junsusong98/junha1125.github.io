---
layout: post
title: 【Pytorch】 TENT & ProDA teardown reports
---

Tent teardown reports 

# 1. Tent

1. `robustbench` & `auto-attack` 설치

   - `pip install git+https://github.com/~~` 방식으로 설치하면, (conda와 충돌이 일어나서 인지) import가 잘 되지 않는다. 
   - Setup.py 내용 수정 `with open("README.md", "r", encoding="utf-8") as fh:`
   - ~~그래서 직접 git clone을 수행한 수, `$ python setup.py install` 을 해주면 잘 설치 된다.~~
   - Avoid using `python setup.py install` use `pip3 install .` [link](https://stackoverflow.com/questions/1550226/python-setup-py-uninstall)
   - **절대 `python setup.py install` 로 설치하지 말기.** 이렇게 설치하면 pip3 uninstall 이 불가능해서, 다른 방법으로 remove 해야한다.
   - ([stackoverflow](https://stackoverflow.com/questions/41060382/using-pip-to-install-packages-to-anaconda-environment/44066694)) 그냥 pip3 ($ which pip3 >> /usr/bin/pip3) 에 설치하면 `pip3 install .` 로 설치해도 condo python이 package를 찾지 못한다. 
   - 따라서, `$ coonda install pip` > `$ which pip > /opt/conda/bin/pip` > `$ pip install .` 순서로 진행해줘야 정상적으로 사용가능!
   - 즉 /usr/bin/pip3 가 아니라 /opt/conda/bin/pip 로 `$ pip list` 와 연동되어야만, `$ conda list` 와 연동이 된다.

2. 계속 뜨는 에러: `AttributeError: 'PosixPath' object has no attribute 'tell'`

   - 이 에러 때문에 conda list와 pip list를 /opt/conda/bin/ 내부에 모아주는 방법을 깨달았다. 하지만
   - 위의 작업을 모두 했는데도, 위의 에러를 해결할 수 없었다. 
   - `$ conda create -n tent python=3.6` 처음부터 이렇게 설치하면 `$ conda install pip` 을 할 필요가 없다.
   - 결국 이렇게 해서 새로운 env에서 처음부터 다시 설치해서, 문제 해결 완료.
     - $ pip install -r requirements.txt
     - $ cd ../robustbench && pip install .
     - $ cd ../auto-attack && pip install .

3. 코드에는 논문의 나온 모든 내용이 나오는게 아니고, cifar10 데이터셋에 대해서만 실험하는 코드가 들어가 있다. 

4. cifar10으로 pre-trained된 network에, Unseen domain으로 adversarial attack이 적용된 cifar10이 들어간다. 

5. Code tree    

   ```sh
   TENT
   |-- cfgs 
   |   |-- norm.yaml
   |   |-- source.yaml
   |   └-- tent.yaml
   |-- conf.py # config load (load_cfg_fom_args)
   |-- norm.py
   |-- tent.py
   └-- cirar10c.py
   ```






## 1.1 [tent.py](https://github.com/DequanWang/tent/blob/master/tent.py) 

```python
import tent

model = TODO_model()

model = tent.configure_model(model)
params, param_names = tent.collect_params(model)
optimizer = TODO_optimizer(params, lr=1e-3)
tented_model = tent.Tent(model, optimizer)

outputs = tented_model(inputs)  # now it infers and adapts!
```

1. `def configure_model`
   - model.train()
   - model 안의 parameter 들의 requires_grad_ 를 false로 모두 바꾸고
   - BatchNorm2d 만 True 처리를 해준다. 
   - 이때 track_running_stats == False 처리는 것 잊지말기. ([참고 코드](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py#L144), [BN document](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html), [what is track_runing](https://discuss.pytorch.org/t/why-track-running-stats-is-not-set-to-false-during-eval/25412))
   - track_running_stats(:bool)이란? γ, β (rescaling 계수)에 대해서 이야기하는게 아니다. standardization 을 위한 channel-wise-mean, channel-wise-std 를 구하기 위해서 momentum update를 할 것인가? 를 의미한다. evaluation에서는 지금까지 momentum update로 구한 channel-wise-mean, channel-wise-std를 사용해서 standarization을 수행한다. 이렇게 하는게 안정적인 prediction을 위한 E(x), V(x)이고 (대신 test 때도 비슷한 domain이 들어올거라는 가정이 필요하다), "γ, β"와 적절히 매칭되는 값들로 standarization을 한다고도 할 수 있다.
   - (지금까지 나 스스로 명명해서 사용하던 모델의 하나하나 layer를, torch에서는 module이라고 표현한다. 그래서 model.modules() 라는 for를 위한 함수도 존재한다.)
2. `def collect_params`
   - NetModule.named_parameters() 을 사용해서 
   - isinstance(m, nn.BatchNorm2d) 인 모듈만을, params, names 을 return 한다. 
   - PS) BatchNorm2d 또한 weight와 bias 변수를 가지고 있고, 이들이 [affine](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) (γ, β) 이다.
3. `clss Tent(nn.Module)`
   - 들어온 이미지에 대해서, BN 학습을 수행한다. 
   - forward 내부에 loss.backward(), optimizer.step() 이 모두 있다. ⭐️
   - 변수
     - [Step](https://github.com/DequanWang/tent/blob/master/tent.py#L31): forward로 들어온 이미지 배치를, step번 iter돌면서 (Entropy를 사용한) BN adapting을 수행한다. 
     - [episodic](https://github.com/DequanWang/tent/blob/master/tent.py#L27): forward를 하기 전에, model load를 할지 안할지. (굳이 true를 할 필요없음. `cifar10c.py` 함수에서 사용)
   - 새로운 decorator
     - @[torch.jit.script](https://pytorch.org/docs/stable/generated/torch.jit.script.html) : code compile 시에 사용하는 거라는데..
     - [@torch.enable_grad()](https://pytorch.org/docs/stable/generated/torch.enable_grad.html) : 함수 안의 연산 중 혹시 no grad 변수가 있으면, 자동으로 requires_grad_ = ture 처리를 해준다.



## 1.2 [norm.py](https://github.com/DequanWang/tent/blob/master/norm.py)

- **Loss를 사용하지 않는다.** (따라서 Error, Loss, Optimizer step과 같은 것은 없다.)
- Normalization 그 자체만으로 ~~학습시킨다.~~ target에 적응하려고 노력한다. (Normalization 코드와 논문을 구체적으로 봐야할 듯 하다. 하지만 대충, unseen domain 이미지들의 feature mean, std를 가지고 [cumulative moving average](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py#L380) 를 적용해서 mean, std 를 정한다고 보면 될 듯 하다. )
- PS) [Batchnorm](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py#L14) 코드에서 self.weight(γ), self.bias(β), self.runing_mean, self.runing_var는 모두 다른 것이다. 
- 특히 self.runing_mean, self.runing_var 변수는 track_running_stats (학습상태를 변수에다가 저장해 둘지 말지) == ture 일때만 남아있는 변수이다. (구체적으로 어떻게 사용하는지는 몰라도 될 것고, 대강 위와 같다고만 알아두자. )
- `def configure_model`
  - tent와 다르게, 처음에 model.requires_grad_(False) 을 하지 않는다. (cifar10c.py 부분에도 model.eval() 부분이 없다. 모델 전체를 train() 모드로 해서 evaluation 하고 있는 듯 하다. 
  - BatchNorm module만을 [m.train()](https://github.com/DequanWang/tent/blob/master/norm.py#L53) 해준다. (굳이 안해줘도 되지만, 혹시 모르니 한번 더 해주는 것 같다. )
  - 추가로 reset_running_stats / track_running_stats=false 설정은 옵션이다.



## 1.3 [cifar10c.py](https://github.com/DequanWang/tent/blob/master/cifar10c.py)

- 위의 함수와 파일들을 사용해서 쉽게, `def evaluation`을 진행한다. 따라서 코드가 매우 심플하다. 
- 몇 Epoch으로 BN update를 할건지는 이 코드에서는 적혀있지 않다. 단, STEP 변수가 1이므로 `def evaluation` 를 할 때 마다, 계속 update가 진행된다. 중간에 STEP=0 으로 바꿔주면 BN 학습은 더이상 이뤄지지 않고, evaluation이 진행된다.
- config파일의 변수 'cfg.CORRUPTION.SEVERITY':  adversaral attack을 얼마나 강하게 줄것인지?  





---

---



# 2. ProDA

Code Link: https://github.com/microsoft/ProDA

논문 설명 Post: [Prototypical Pseudo Label Denoising and Target Structure](https://junha1125.github.io/blog/artificial-intelligence/2021-04-02-ProDA/)

Directory Tree

```sh
ProDA
|-- calc_prototype.py   				# protopype 를 계산해서 찾아주는 함수
|-- data
|   |-- DataProvider.py   			# 3개의 dataset loader 중 하나를 최종 return해주는 함수
|   |-- __init__.py
|   |-- augmentations.py				# augmentation
|   |-- randaugment.py
|   |-- base_dataset.py  				# 3개의 dataset loader의 뿌리
|   |-- cityscapes_dataset.py
|   |-- gta5_dataset.py
|   └-- synthia_dataset.py
|-- generate_pseudo_label.py		# 지금까지 학습된 pretrained-parameter를 가지고 pseudo_label 임시로 만들어 줌
|-- models
|   |-- adaptation_modelv2.py
|   |-- deeplabv2.py
|   |-- discriminator.py
|   |-- sync_batchnorm
|   └-- utils.py
|-- metrics.py									# score를 계산해서 return 해준다
|-- parser_train.py							# parser.add_argument
|-- utils.py										# logger and fliplr
|-- train.py
└-- test.py
```

학습 과정 및 추론 과정 

1. Pseudo label을 생성해야 하기 때문에, 몇가지 Step을 거쳐야 한다. 따라서 학습과정이 매우 복잡하다.
2. [학습과정 순서](https://github.com/microsoft/ProDA/blob/main/README.md#training)
   - Stage1
     - `generate_pseudo_label.py` >> soft pseudo label 만들기
     - `calc_prototype.py` >> momentum update를 사용해서 Prototype(centroids) 1개만 계산해 놓는다.
     - `train.py` >> moving_prototype는 물론이고, [Total loss](https://junha1125.github.io/blog/artificial-intelligence/2021-04-02-ProDA/#42-structure-learning-by-enforcing-consistency) 의 모든 연산을 적용한다. (structure learning & regular)
   - Stage2: `knowledge distillation` 를 적용해서 SimCLR으로 init된 모델을 다시 학습시킨다. 
     - `generate_pseudo_label.py` >> flip까지 사용해서, pseudo label 만들기
     - `train.py` >> moving_prototype는 더 이상 안하고, 위에서 미리 생성된 pseudo_label을 사용해서 distillation 으로 학습 수행
   - Stage3: Stage2 반복
     - `generate_pseudo_label.py` >> (위와 동일) + bn_clr 사용
     - `train.py` >> (위와 동일) + ema_bn 사용
3. [추론과정 순서](https://github.com/microsoft/ProDA/blob/main/README.md#inference-using-pretrained-model)
4. [Parser argument 모음](https://github.com/microsoft/ProDA/blob/main/parser_train.py)

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





## 2.2 source only 학습시키기 위한 여정

1. dataset 문제

   1. gta5 : split.mat 파일 다운받아야함
   2. synthia : 데이터셋 잘못 받음

2. ```python
   def train(opt, logger):
   	if opt.stage == 'warm_up':
     elif opt.stage == 'stage1':  
     elif opt.stage == 'src_only':
     	loss_GTA = model.step_source(images, labels, source_imageS, source_params)
   
   # ProDA/models/adaptation_modelv2.py 
   # step_adv 에서 Adversarial learning & Discriminator 만 제거하면 돼서, 아래와 같이 쉽게 만들었다. 
   class CustomModel()
   	def def step_source(self, source_x, source_label, source_imageS, source_params):
   ```

3. ```sh
   $ python /workspace/ttt_memory/ProDA/train.py --config '/workspace/ttt_memory/ProDA/configs/src_only_cityscapes.py'
   $ python /workspace/ttt_memory/ProDA/train.py --config '/workspace/ttt_memory/ProDA/configs/src_only_synthia.py'
   $ python /workspace/ttt_memory/ProDA/train.py --config '/workspace/ttt_memory/ProDA/configs/src_only_gta5.py''
   ```

4. Source only에서 몇개의 클래스에서 성능이 낮게 나온이유

   - 지금까지 iter만 있으면 되지, Epoch이 이제는 무슨 쓸모지? 라고 생각했다. 하지만 Epoch은 전체 데이터셋을 한바퀴 다 봤다는 의미이므로 몇 Epoch으로 데이터셋을 학습했는지도 굉장히 중요한 지표이다. 얘를들어서 GTA5는 2만개의 dataset이 존재한다. 2만 iter를 돌았다고 하더라도 겨우 1epoch만 돌은 것이다. 이런 경우 적은 클래스를 가지는 객체는 당연히 학습이 잘 안될 수 있다. 
   - 더군다나, AdpatSeg에서는 resize로 데이터를 확장하고 crop을 하지 않는다. only [1024, 512] crop만을 진행한다. 반면에 ProDA에서는 2200 resize를 하고 [812, 512] crop을 진행한다. 이러니.. 드믈게 존재하는 class pixel에 대해서 학습이 더욱더 안될만하다. 
   - 이러한 점들이 내가 남긴 이 Issue에 대한 답변이라고 할 수 있고, 내가 해야하는 것은! 학습을 더 많이 해보는 것이다!



## 2.3 args -> conf

- https://github.com/open-mmlab/mmdetection/blob/master/tools/train.py

- https://github.com/open-mmlab/mmcv/blob/d9effbd1d0da393b46ee4524e8ce8f52245e9bba/mmcv/utils/config.py

- ```sh
  wget https://raw.githubusercontent.com/open-mmlab/mmcv/master/mmcv/utils/config.py
  wget https://raw.githubusercontent.com/open-mmlab/mmcv/master/mmcv/utils/misc.py
  wget https://raw.githubusercontent.com/open-mmlab/mmcv/master/mmcv/utils/path.py
  pip install addict
  pip install yapf
  ```

- ```python
  ## read conf and merge with parse
  opt = Config.fromfile(args.config)
  # opt.merge_from_dict(vars(args))
  print(f'Config:\n{opt.pretty_text}\n=========== Start Traning ===========')
  ```



## 2.4 logger 

print 및 내용 저장에 좋으니, 나중에 참고해서 사용하자



## 2.5 debug

1. 뭔가 안된다면, worker = 1 로 해놓고 디버깅 할 것.
2. 왜 안되는지 앞으로 따라가 보고, 앞으로 따라가도 모르겠으면
3. 뒤로 가야한다. 뒤로가서 무엇을 하고 나서 이상이 생긴건지. 
