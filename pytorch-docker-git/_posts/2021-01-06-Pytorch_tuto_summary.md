---
layout: post
title: 【Pytorch】Pytorch Tutorial 내용 핵심 정리
# description: >
---

- Pytorch Tutorial 내용을 핵심만 요약 정리 하였습니다.
- 저 코드와, 한줄한줄 공부한 내용은 `/sb_kookmin_googledrive/GCPcode/pytorch` 에 있습니다.

# 1. pytorch document 순서 정리
1. torch - create, indexing, Math Operation 
1. torch.Tensor - details on the above
1. torch.autograd - backword, grad
1. torch.nn - layer, loss
1. torchvision - dataset
1. torch.utils.data - Dataloader
1. torch.optim
1. torch.utils.tensorboard
1. torchvision.transforms - data augm
1. torchvision.models

# 3NeuralNetworks in 60min learn
1. Numpy를 이용한 가중치 갱신해보기
    - Numpy 함수들을 이용해서 2층 affine Layer 구성하기
2. torch.tensor를 사용해서 2층 layer 구성하기
    - Numpy와 거의 동일하지만 매소드 함수 이름이 가끔 다름
3. Autograd
    - backward를 손수 chane법칙으로 계산할 필요없다. 
    - loss.backward() 해버리면 끝!
    - 갱신은 w2 -= learning_rate * w2.grad
    - [torch.tensor.autograd](https://pytorch.org/docs/stable/autograd.html#tensor-autograd-functions) : grad, required_grad 맴버변수 있음
    - [nn.Module 클래스의 parameters에 대한 고찰](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.parameters) 
4. 새로운 Layer, Function 정의하기
    - torch.clamp라는 함수를 사용하면, relu 처럼 동작 가능 + loss.backward할 때 backward알아서 처리 됨.
    - 하지만 직접 relu를 정의하면?? backward가 안된다. 
    - torch.autograd.Function 상속하는 class를 만들어야 한다. ([torch.autograd.Function](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function))
5. nn 모듈 - Sequential과 Module과 같은 container 존재
    - 정의 방법과 사용방법이 다르다. 
    - 하지만 zero_grad(). backward(), parameters(), step() 사용법은 모두 같다. 
    - 갱신은 param -= learning_rate * param.grd (간단한 고찰 있음)
6. Optimiaer
    - 갱신은 Optimizer.step()! 
7. 같은 원리로, resnet을 만드는 건 쉽다.
    - model.foward 만 잘 손보면 된다. 

# 4Classiffier.ipynb in 60min learn
1. Data Load 하기
	- torchvision.transform
	- trainset = torchvision.datasets
	- torch.utils.data.DataLoader
2. 신경망 정의하기
	- class (torch.nn.module)
	- \_\_init\_\_, forward
3. Loss, Optimizer 정의
	- criterion/loss = torch.nn.LOSS_FUNC  
	- optimizer = torch.optim.OPTIM_FUNC
4. 신경망 학습 시키기
	- for epoch
	  - for i, data in enumerate(trainloader) # data = \[label, img\]
	    - optimizer.zero_grad()
	    - outputs = net(inputs)
	    - loss = criterion(outputs, labels)
	    - loss.backward()
        - optimizer.step()
5. Test하기
	- outputs = net(images)
	- with torch.no_grad():
	  - for data in testloader: 
	    - outputs = net(images)
	    - _, predicted = torch.max(outputs.data, 1) 
	    - correct += (predicted == labels).sum().item()
	- class별 accuracy 분석은 코드 참조
6. 신경망 저장, load하기
	- [torch.save](https://pytorch.org/docs/stable/generated/torch.save.html#torch.save) # 신경망 저장, 변수 저장 모두 가능, .pt 파일
	- nn.module.state_dirt()
7. GPU 사용하기
	- net.to(device)
	- data\[0].to(device), data\[1].to(device) # label, data

# 3.WhatIsTorch.nnReally.ipynb
깨달은점들  
- [Torch basic operation 정리 해놓은 사이트](https://jhui.github.io/2018/02/09/PyTorch-Basic-operations/)
- 그냥 단순 무식하게 함수 찾아서 공부하지 말자. 예제 코드를 보면서 그 코드에서 어떻게 사용되고 있는지를 보고, 공부하자. 예를 들어서, 지금 TensorDataset이 뭔지 모른다. 하지만 지금 알 필요는 없다. 나중에 Detection, Segmentation 코드 공부할 때, 나오면 깨달으면 된다.

1. MNIST Data download
    - 새로운 모듈
        - from pathlib import Path = os.path
        - import requests = wegt역할
        - import pickle = File loading
        - import gzip = 압축 안 풀고 읽기
    - map(torch.tensor, *datas)
2. tensor만 사용해서 신경망 구현
    - loss = -input[range(target.shape[0]), target].mean() # 배월 원소 특정 부분
    - preds = torch.argmax(preds_before, dim=1) 
3. Using torch.nn.functional
    - loss_func = F.cross_entropy
4. Refactor using nn.Module
    - class Mnist_Logistic(nn.Module)
    - self.weights = nn.Parameter(torch.randn(784, 10)
    - for p in model.parameters(): 
5. Refactor using nn.Linear
6. Refactor using optim
7. Refactor using Dataset
    - torch.utils.data.TensorDataset
8. Refactor using DataLoader
    - train_ds = TensorDataset(x_train, y_train)
    - train_dl = DataLoader(train_ds, batch_size=bs)
    - for (xb, y) in train_dl:
9. Add validation
    - model.eval()
    - with torch.no_grad(): valid_loss 출력해보기 
10. 지금까지 했던 것을 함수로 만들기!
    - for epoch in range(10):
        - model.train()
        - for xb, yb in train_dl:
        - model.eval()
        - with torch.no_grad():
11. nn.linear말고, CNN 사용하기
    - def forward(self, xb):  
        - xb = xb.view(-1, 1, 28, 28)
        - return xb.view(-1, xb.size(1))
12. nn.Sequential
    - nn.Sequential에는 nn.Module을 상속하는 layer들이 들어가야 함.
    (Ex. nn.Conv2d, nn.ReLU() 등 init에서 사용하는 함수들) 
13. Wrapping DataLoader
    - 모델에서 데이터 전처리 하지 말고. 전처리한 데이터를 모델에 넣기 위해.
    - train_dl = WrappedDataLoader(train_dl, preprocess)
    - class WrappedDataLoader 구현
    - def \_\_iter\_\_(self): batches = iter(self.dl); for b in batches: yield (self.func(*b))
14. Using GPU
    - model.to(dev)
    - Input data.to(dev)
    - 항상 이 2개!

# 4.tensorbord.ipynb ⭐
- 참고 사이트 : [tensorboard documentary](https://pytorch.org/docs/stable/tensorboard.html#torch-utils-tensorboard) 
- torch_module_research.ipynb 파일도 꼭 같이 공부하기

1. DataLoader, Net, Loss, Optim - Define
2. writer = torch.utils.tensorboard.SummaryWriter()
3. DataLoader
    - DataLoader는 \_\_iter\_\_가 잘 정의된 하나의 단순 class일 뿐이다. 
        -"3.what torch.nn"에서 WrappedDataLoader 도 참조
    - DataLoader사용 방법은 3가지 이다. 
        1. for x,y in DataLoader:  
        2. x, y = iter(DataLoader).next() 
        3. x, y = next(iter(DataLoader))
        4. PS: return되는 type은 list\[ x_torch.tensor_batch, y_torch.tensor_batch ] 이다. 
4. torch.utils.tensorboard.SummaryWriter()의 대표 매소드 함수
    - writer.add_image("title=tag",img_grid) : input img_grid 보기
        - img_grid = torchvision.utils.make_grid( torch.tensor( 몇장, depth, w, h) ) : '몇장' 에 대해서 grid_image를 만들어 tensor로 뱉어줌.(torch_module_research.ipynb 꼭 참조) 
    - writer.add_graph(model,input_data) : model Architecture 보기
    - writer.add_embedding() : 고차원 Data 3차원으로 Visualization 하기
5. Train
    - torch.max(arr, dim= 제거할 차원)  
    - np.squeeze(np.arry) : shape에서 1은 다 없애 버린다. 2차원으로 감지되는 행렬을, 1차원으로 바꿔주는 역할도 한다.
    - torch.shape, np.shape : if 2차원이라고 감지되고 있다면, (1,4)와 (4,1)은 완전히 다른거다. 
    - torch.nn.functional.softmax(input, dim=None) : Ex) dim=1 일 때 arr\[1,:,2] = 1차원! 이 1차원 백터에 대한 모든 원소 softmax
    - 학습 도중 writer.add_scalar, writer.add_figure
        ```python
        for epoch in range(2):  
            for i, data in enumerate(trainloader, 0):
                if i % 1000 == 0:
                    writer.add_scalar("title=tag", scalar_value, global_step) - Loss plot
                    writer.add_figure
        ```
6. Test
    - torch.cat : concatenation
    - torch.stack : list to tensor
    - 각 class에 대한 precision-recall curve 그리기
        - writer.add_pr_curve(class이름, TrueOrFalse, PositiveOrNagative)

# 5.torchvision_finetuning_instance_segmentation.ipynb
- 6.transfer_learning_tutorial.ipynb를 먼저 공부하고, 여기 보는게 낫다.

1. 새로운 Dataset 정의하는 방법
    - torch.utils.data.Dataset을 상속하는 클래스. 
    - \_\_len\_\_ and **\_\_getitem\_\_** 정의하기
    - \_\_getitem\_\_ (self, idx) 은 \[image, target\]을 return해야한다.
    - 자세한 내용은 파일 참조
2. Dataset을 새로 정의하는 Class 만들기
    - os.path.join, os.listdir
    - target["masks"]를 정의하기 위한 변수 만들기
        - broadcastion 적용 : masks = (mask == obj_ids[:, None, None]) 
3. torchvision.models를 사용한 model 정의하기
    - 아래의 내용은 필요할 때! Tuto 코드 & Torch Git 코드 꼭 봐야 함. 
    1. torch Git, torchvision Git의 함수나 클래스 호출하는 방법 (헷갈리면 읽어보기!)
    1. model의 마지막 단에 나오는 class 갯수만 바꾸고 싶을 때.
    1. model의 class갯수와 backbone network를 모두 바꾸고 싶을 때
    1. 2번 방법으로 get_model_instance_segmentation함수 정의하기.
4. train과 evaluation 하기
    1. dataset = PennFudanDataset(위에서 내가 만든 클래스)
    
    1. 하나의 데이터를, train set, validation set으로 나누는 방법     
        ```python
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-50])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
        ```
        
    1. data_loader = torch.utils.data.DataLoader(dataset, batch_size=2 ... )
    
    1. model.to(device)
    
    1. optimizer정의 후 lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer ...) 
    
        - optimizer.step,() 후 lr_scheduler.step() 또한 해줘야 함.
    
    1. epoch 돌면서 Git의 vision.reference.detection.engine의 train_one_epoch, evalate 사용해서 log 출력

# 6.transfer_learning_tutorial.ipynb ⭐
1. Import Modules, Dataloader Define
    - datasets.ImageFolder(root, transforms)
    - dictionary구조를 이용한, datasets, dataloaders 정의 -Ex) dataloaders['train'], dataloaders['test']
2. dataloader가 잘 정의 되었나 확인
    
    - torchvision.utils.make_grid 를 완벽하게 사용하는 방법 - imshow 함수 정의(np.transpose, 정규화, np.clip, show)
3. **Train과 Validation 전체를 하는 함수 정의**
    
    - def train_model(model, criterion, optimizer, scheduler, num_epochs=25):  
        - return model.load_state_dict(best_model_wts)
    - best_model_wts = copy.deepcopy(model.state_dict())
    - time_elapsed = time.time() - since
    - 전체 순서 정리(원본은 ipynb 파일 참조)  
        ![image](https://user-images.githubusercontent.com/46951365/103886436-d76f4c00-5124-11eb-80c6-6b8bc801ee7d.png)
4. Define Function (visualizing the model predictions)
    - def visualize_model(model, num_images=6):
        - ax = plt.subplot(num_images//2 , 2, images_so_far )
5. model 정의하고, Train 및 validation 해보기
    1. Finetuning the conv_Net
        - 직접 torch git 에서 전체 코드를 꼭 봐야 함.
        - 요약  
            ```Python
            model_ft = models.resnet18(pretrained=True)
            model_ft.fc = nn.Linear(num_in_features, 2)  
            model_ft = train_model(model_ft, criterion, optimizer_ft, ...)
            visualize_model(model_ft)
            ```
    2. Conv_Net as fixed feature extractor
        - 요약  
            ```Python
            model_conv = torchvision.models.resnet18(pretrained=True)
            for param in model_conv.parameters():
                param.requires_grad = False
            num_ftrs = model_conv.fc.in_features
            model_conv.fc = nn.Linear(num_ftrs, 2)
            model_conv = train_model(model_conv, criterion, optimizer_ft, ...)
            visualize_model(model_conv)
            ```

