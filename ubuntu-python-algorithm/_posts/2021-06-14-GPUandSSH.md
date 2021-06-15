---
layout: post
title: 【ubuntu】 GPU 할당 변경 및 SSH 및 Tensorboard 팁
---

**Reference**

1. GPU 할당변경 : [GPU Allocation 변경하기](https://jeongwookie.github.io/2020/03/24/200324-pytorch-cuda-gpu-allocate/)
2. Tensorboard and SSH



# GPU 할당 변경

```python
import torch

# 현재 Setup 되어있는 device 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))

# GPU 할당 변경하기
GPU_NUM = 0 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check

# Additional Infos
if device.type == 'cuda':
    print(torch.cuda.get_device_name(GPU_NUM))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(GPU_NUM)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(GPU_NUM)/1024**3,1), 'GB')
```





# SSH 및 Tensorboard

SSH 서버 상태에서 Tensorboard를 보는 방법은 2개다.

1. SSH 설정에서 6006과 같은 특정 포트 열어두기.
2. 서버 6006포트가 locallhost:8808 과 같이 현재 나의 컴퓨터 포트로 연결되도록 설정한다.



2번째 방법을 사용하는 법은 아래와 같다. 

```sh
$ ssh -L localhost:8898:localhost:6009 juncc@143.248.38.123

## 해당포트가 사용되고 있는지 아닌지 확인. 목록이 뜬다면 뭔가 연결되어 있다는 것.
$ lsof  -ti:8898

## 만약 해당포트가 사용되고 있다면, 모든 연결 끊기.
$ lsof -ti:8898 | xargs kill -9
```

현재 나의 컴퓨터 8898 <<--- 서버 컴퓨터의 6009포트



# Tensorboard 명령어 안 먹힐때

Shell에서 Tensorbord라고 치면... 

`Tensorboard command not found` 라고 나온다. 

이럴 때는 해당 사이트([stackover flow](https://stackoverflow.com/questions/45095820/tensorboard-command-not-found))의 답변을 사용해서 아래의 순서대로 실행하면 된다. 

```sh
# Tensorboard 어딘가 설치되어 있는지 확인
$ pip show tensorflow

# Tensorboard 설치
$ pip install tensorflow

# 터미널 자체에서 Tensorboard 명령어가 안먹히면, 약간 돌려서 실행
$ python3 -m tensorboard.main --logdir=~/my/training/dir --port=6006

# 이는 이 파일을 실행하는 것과 같다
$ cd /home/abc/xy/.local/lib/python2.7/site-packages/tensorboard
$ python main.py --logdir=/path/to/log_file/
```

