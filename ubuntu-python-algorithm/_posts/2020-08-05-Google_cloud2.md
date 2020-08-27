---
layout: post
title: 【GPU_Server】 Google Cloud 플랫폼 - 적절한 conda 환경 Tensorflow, Keras 버전 맞추기
description: > 
    딥러닝을 위한 적절한 아나콘다 환경 셋팅하기
---
**이전 게시물을 보고 오시는 것을 추천드립니다.**

## 1. Tensorflow & Keras 버전 세팅 방법
- 참고자료
    <img src="https://user-images.githubusercontent.com/46951365/91464200-41e6c480-e8c7-11ea-81e4-bca640d8c138.png" alt="image" style="zoom:75%;" />
- 추천하는 3가지 가상환경 :   
    1. tf113 : Tensorflow 1.13, Keras 2.2  
    2. tf115 : Tensorflow 1.15, Keras 2.3  
    2. tf_obj : Tensorflow 1.15, Keras 2.3  -> Tensorflow object dection API 사용하기 위한 Env
- 아래의 코드 참고할 것 
    ```sh
    $ conda create -n tf113 python=3.6
    $ conda activate tf113
    $ cd envs/tf113/lib   -> python3.6 설치
    $ cd ./site-packages  -> 기본 package들이 설치되어 있다. 
    $ cd ~/DLCV/data/util/
    $ chmod +x *.sh
    $ ./install_tf113.sh
    ```
- install_tf.sh 내부에 있는 코드 중 $ conda install [nb_conda](https://github.com/Anaconda-Platform/nb_conda) -y 명령어를 통해서 콘다 환경들에게 별도로 jupyter를 따로 실행할 수 있게 해준다.   
- 