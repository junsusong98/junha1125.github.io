---
layout: post
title: 【GPU_Server】 Google Cloud 플랫폼 - VM instance Jupyter로 이용하기
description: > 
    무료로 150시간 정도 편리하게 GPU를 사용할 수 있는 Google Cloud 딥러닝 플렛폼 활용 방법
---
Google Cloud 딥러닝 플렛폼 이용하기

## Google Cloud   VS   Colab
- Colab을 사용해도 좋지만, Google Cloud를 사용하는 것이 빠르고 편리하므로 이것을 사용하는 방법에 대해서 공부할 계획이다. 
- 다양한 데이터를 사용해서, 다양한 모델을 사용해서 공부해볼 예정이다. 

## 1. Google cloud platform 가입하고 활용하기
- google cloud platform [사이트 바로가기](https://cloud.google.com/?hl=ko)
- 콘솔 -> 서비스 약관 확인하기
- 무료로 가입하기 -> 서비스 약관 확인하기
- 무료계정이면 GPU를 사용할 수 없다. 자동 결제를 등록해서 유료계정으로 만들어야 GPU사용가능
- Computer Engine -> VM 인스턴드 생성하기. 설정은 아래와 같이.  
    <img src="https://user-images.githubusercontent.com/46951365/91270459-9dc02900-e7b3-11ea-828e-9146da2d4318.png" alt="image" style="zoom:50%;" />
- 유료 계정으로 업데이트 -> GPU 할당 메일 보니게 -> GPU할당하기
- 다시 VM 인스턴트 생성하기 - T4(추천), P100과 같은 GPU할당하는 인스턴트로 생성하기(운영체제 : 딥러닝 리눅스) -> 인스탄스 생성 안됨 -> 메일 보내야 함. -> IAM 및 관리자 - 할당량 - GPUs all region - 할당량 수정 - 메일 보내기 - 확인 답변 받고 인스턴트 재생성 (최근 메일에 대한 GPU 할당 거절이 많다. CPU서버를 48시간 이상 가동 후 요청을 다시 해보는 것을 추천한다.)  
    <img src="https://user-images.githubusercontent.com/46951365/91290793-0b2d8300-e7cf-11ea-9e64-c3e590bdccf8.png" alt="image" style="zoom:50%;" />  
- 우선 Colab에서 작업하고 있자. Colab에서 나의 google drive와 마운트하고 그리고 작업을 수행하자. 코랩 오래 할당시 주의할 점 ([런타임 연결 끊김 방지 정보](https://bryan7.tistory.com/1077))  
- 



## 2. Google cloud platform 이란. + Port Open하기. 
- 다음의 동영상([Youtube링크](https://www.youtube.com/watch?v=z6WOMYI-WiU&list=PL1jdJcP6uQttVWZTd1X2x22kv32Rhkiyc))을 보고 공부한 내용을 기록합니다, 
1. 구글 클라우드플랫폼 입문 활용  
    - 아마존 AWS와 MS Azure 서비스와 비슷한 서비스이다. 하지만 경쟁사 서비스보다 늦게 시작했기 때문에 가성비가 좋다. 그리고 용어 하나하나가 AWS보다 이해가 쉽다는 장점이 있다. 
    - 프로젝트 단위로 움직이다. 프로젝트당 Computers, VM 인스턴스들, Docker들을 묶음으로 사용할 수 있다. 프로젝트를 크게 한다고 하면 한달에 40만원이면 아주 편하게 사용할 수 있기 때문에 매우 유용하다. 
    - 컴퓨팅 - 컴퓨터 엔진, 쿠버네틱스 엔진, 클라우드 Function, 클라우드 Run 등 가장 중요한 요소들이 존재한다. 
    - 이 중 컴퓨터 엔진의 VM instance를 가장 많이 사용하게 되고 이 인스턴트 하나가 가상 컴퓨터 하나(하나의 기본적은 자원)라고 생각하면 편하다. 
    - VM instance : Region(GPU, CPU 하드웨어가 있는 지역), 영역, 시리즈, 용도들을 선택하면서 가격이 변화하는 것을 확인할 수 있다. GPU를 사용하면 비용이 많이 든다. Colab에서는 무료로 GPU를 사용할 수 있기 때문에 Colab을 사용하는 것이 유용하다.
    - VM instance 설정 :  엑세스 범위를 설정해서 인스턴트 끼리의 연결, 공유를 가능하며 방화벽 설정은 무조건적으로 체크하는 것이 좋다. (나중에 바꿀 수 있다) 보안, 가용성 정책을 사용하면 가격이 저렴해지나 리소스가 전체적으로 부족해지면 리소스가 뺏길 수도 있다. 
    - VM instance 생성 완료 : 외부 IP는 가변적이다. 알아두자. 내부 IP는 같은 Region의 인스턴트끼리 소통할 수 있는 IP이다. 

2. 인스턴트 SSH 접속과 도커 서비스, 방화벽 규칙
    - SSH에 접근하는 방법은 SSH를 클릭하거나, 다른 외부 SSH에서 접속하기, 웹으로 접속하기 3가지 방법이 있다. 아래의 방법을 통해서 웹으로 접속 설정을 해보자.
    - SSH 접속을 통해서 우분투와 접근할 수 있다. SSH-브라우저창 열기를 하면 Terminal을 쉽게 열 수 있다. apache2를 설치할 것(80/TCP로 웹서비스 사용 가능). git을 설치할 것.

    - ```sh   
      $ sudo apt update
      $ sudo apt-get install apache2
      $ sudo service apache2 start
      $ sudo apt-get install git
      $ sudo netstat -na | grep 80
      ```
    - 외부 IP 클릭! -> 주소창 http 지우기 -> Apache2 Debian Default Page 확인 가능.
      Apache2를 그대로 사용하지 않고 Docker환경을 이용해서 어플리케이션 올릴 예정
    - Docker설치가 안되면 https://docs.docker.com/engine/install/debian/ 여기서! 우분투 아니라 데비안이다! 그리고 apt-get update에서 docker fetch 문제가 있어도 " sudo apt-get install docker-ce docker-ce-cli containerd.io " 걍 해도 잘 동작하는 것 확인 가능. 

      ```sh
        $ sudo apt install docker.io

        $ sudo docker search bwapp
        $ sudo docker pull raesene/bwapp (또는 wordpress)
        $ sudo docker run -d --name webconnet -p 81:80 raesene/bwap
      ```

    - 이제 웹에서 외부 IP:81 로 들어가준다. 하지만 들어가지지 않은 것을 확인할 수 있다. 이것은 구글 클라우드에서 방화벽 처리를 해놓았기 때문이다. 
    - Google Cloud Platform - VPC 네트워크 - 방화벽 - 방화벽 규칙 만들기 - 소스 IP 범위 : 0.0.0.0/0  tcp: 81 정의 - 그리고 다시 외부 IP:81로 접근하면 아래와 같은 화면 결과.  
      <img src="https://user-images.githubusercontent.com/46951365/91317607-3aef8180-e7f5-11ea-8461-c5753a54973a.png" alt="image" style="zoom:67%;" />  
    - 외부 IP:81/install.php 로 들어가면 bwapp(웹해킹)사이트로 들어갈 수 있다. 

    - 나는 이 방법을 사용해서 8080 port의 방화벽을 허용해 놓고, ml-workspace Image를 가져와서 Container를 실행했다. 다음과 같은 명령어를 사용했다.

      ```sh
      $ sudo docker run -d \
      -p 8080:8080 \
      --name "ml-workspace" -v "${PWD}:/workspace" \
      --env AUTHENTICATE_VIA_JUPYTER="eoqkr" \
      --restart always \
      mltooling/ml-workspace:latest
      ```

    3. GCP의 MarektPlace 활용하기. 저장 디시크 활용
      - 추가적인 강의는 추후에 들을 예정.
      - [추가 강의 링크](https://www.youtube.com/watch?v=8ld759re0Xg)


## 3. 딥러닝용 가상환경 구축하기
- 1에서 GPU 할당을 받았다 그 후 작업하는 과정을 요약해 놓는다.   
- 아래의 과정은 SSH 크롬 브라우저를 사용해도 되지만, 고정 IP를 사용해서 Putty로 쉽게 윈도우에서 연결할 수 있도록 설정하는 방법을 적어 놓은 것이다. (Google cloud vm instance Putty connect 등으로 구글링하면 나올 내용들을 정리해 놓는다.)
1. 4코어 15GB 메모리. GPU T4 1개. 부팅디스크 (Deep Learning on Linux) 디스크 300GB. 
2. Cloud 서버 활용하기 - IP설정(고정적으로 만들기) : VPC 네트워크 - 외부 IP주소 - 고정 주소 예약 (region-연결대상 만 맞춰주고 나머지는 Default로 저장)
3. 방화벽 규칙 - 방화벽 규칙 만들기 - 8888 port 열기 - 네트워크의 모든 인스턴스 - tcp:8888 -> 
4. Putty donwnload - Putty gen (Private Key 만드는데 사용) 열기 - 인스턴스 세부정보 수정 - Key generate(Key comment : 구글 계정 아이디 넣기) - Key 복사해서 SSH 키 입력 -  (Putty gen) Save private, public key - VM 인스턴스 세부정보 수정 저장. 
5. 외부 IP를 Putty에 넣어주고, SSH Auth Browsd - 위에 저장한 Private key 클릭 - Host Name을 sb020518@외부IP 로 저장하고 save_sessions - Open.
6. Nvidia driver 설치 Yes. - Nvidia driver installed - nvidia-smi 체크해보기
7. Nvidia driver가 잘 설치되지 않는다면 다음의 과정을 거친다.  

    ```sh
    $ cd /opt/deeplearning
    $ sudo ./install-driver.sh
    ```

## 4. 주피터 노트북 Setup 하기
- 실습을 위한 코드를 다운 받고, 아나콘다를 설치 후. Jupyter server를 설치한다. 
- 아래의 과정을 순서대로 수행하면 된다.  

    ```sh
    $ git clone ~~culminkw/DLCV
    - anaconda download 하기.(wget 링크주소복사 및 붙여넣기)
    $ chmod 777 ./Anaconda3
    - 콘다 설치 완료
    $ cd ~/
    $ jupyter notebook --generate-config
    $ cd ~/.jupyter
    $ vi ~/.vimrc   ->   syntax off  -> :wq!  (편집창 색깔 이쁘게)
    $ vi ~/.jupyter/.jupyter*.py   ->   DLCV/data/util/jupyer_notebook_config.py 의 내용 복붙 해놓기 
    - (차이점을 비교해서 뭐가 다른지 공부하는 시간가지기 Ex.외부포트 공개, 비밀번호 없음 등... )
    $ cd && vi start_jn.sh    ->  nohup jupyter notebook &  (back End에서 실행)
    $ chmod +x start_jn.sh
    $ ./start_jn.sh
    $ tail -f nohup.out   (jupyter 실행라인이 보여야 함)
    - http:// VM instance 외부-IP:8888  (https 아님)
    - jupyter 실행되는 것을 볼 수 있다. 
    ```   
    <img src="https://user-images.githubusercontent.com/46951365/91463123-fbdd3100-e8c5-11ea-9853-dd572f5c6eb4.png" alt="image" style="zoom:67%;" />    

- 이와 같이 우리의 SSH 환경이 jupyter에서 실행되는 것을 확인할 수 있다. 

## 5. GCP tensorboard 열기
- GCP의 SSH를 이용하는 경우
    1. SSH를 GCP를 통해서 열어서 $ tensorboard --logdir=runs 실행
    2. TensorBoard 2.2.1 at http://localhost:6006/ (Press CTRL+C to quit)  
      라고 나오면 이떄 나오는 링크를 클릭해서 들어가야 tensorboard를 볼 수 있다. 
    3. [참고했던 사이트](https://www.montefischer.com/2020/02/20/tensorboard-with-gcp.html) 이 작업을 해서 되는건지 아닌지는 모르겠다. 이거 했을 때 에러가 많았는데...
    3. 인스턴스_외부_IP:6006/ 와 같은 링크로 크롬에 직접 쳐서는 들어갈 수 없다.

- jupyter notebook의 terminal를 이용하는 경우
    1. 이상하게 jupyter notebook을 틀어서 위와 같은 작업을 똑같이 실행하면
        - localhost:6006/로 들어갈 수 없다
        - 실행하던 terminal이 멈처버린다. ctrl+c도 안 먹힌다. 
        - terminal을 shortdown할 수도 없다. 
    2. 내 생각인데, 인스턴스_외부_IP:8888에서 다시 6006을 열라고 하니까 안되는 것 같다. 
        - 주인이 있는데, 고객이 고객을 상대하려고 하는 꼴인건가??
    3. 옛날에 은환이가 하던 jupyter 메뉴에서 tensorboard를 여는 방법은, 은환이 왈, 'jupyter notebook에서 tensorboard와 연동이 안되어 있을 거다.' 나중에 jupyter에서 tensorboard를 여는 방법은 연동을 해서 할 수 있도록 찾아보자. ([ml_workspace](https://github.com/ml-tooling/ml-workspace#tensorboard) 에서 할 수 있는 것처럼.)
    4. jupyter로 작업하는 동안, vscode와 GCP로 tensorboard를 열려고 하면 event파일을 다른 프로세서에서 잡고 있기 때문에 열 수 없다. 따라서, 주피터 cell에다가 ! tensorboard --logdir=runs 를 치고 localhost로 들어가면 된다. 신기하게 여기서 localhost도 내 노트북 IP를 사용한다.
    5. 근데 4번처럼 하면 다른 셀을 실행할 수가 없다. 개같다...

- vscode를 이용하는 경우
    1. terminal에서 $ tensorboard --logdir=runs 실행하면, localhost:6006링크를 준다. 
    2. 그리고 그 링크를 클릭하면, 나의 노트북 ip를 이용해서(신기방기) tensorboard를 열어준다!!
    3. 이게 GCP SSH 보다 훨씬 편할 듯 하다! 

- 최종방법! 
    1. 다음 사이트를 참고 했다. [jupyter에서 tensorboard 사용법](https://lucycle.tistory.com/274)
    2. 'vision' conda env에서 pip install jupyter-tensorboard 실행(conda install 없다.)
    3. 성공!  
      ![image](https://user-images.githubusercontent.com/46951365/96361914-98081380-1164-11eb-9cc2-7c969820a211.png)
    

## 6. GPU 할당 받은 instance 만들기
1. GPU 할당은 받았다. 하지만 Credit을 사용하고 있어서 GPU instance 만들기 불가능
2. $ gcloud computer 를 사용하는, google console 을 사용해보기로 했다. 
3. 다음과 같은 명령어를 사용했다. 걸리는 점은 image project와 family설정을 아무리 해줘도, cuda가 이미 잘 설치된 image는 설정이 불가능하더라. [GCP image document](https://cloud.google.com/compute/docs/images#image_families)   

  ```sh
    gcloud compute instances create p100-1 \
    --machine-type e2-standard-2 --zone us-west1-b \
    --accelerator type=nvidia-tesla-p100,count=1 \
    --image-family debian-9  --image-project debian-cloud   \
     --restart-on-failure 
    # [--preemptible] 다른 사람이 사용하면 나의 GPU를 뺏기게 가능하게 설정 But 저렴한 가격으로 GPU 사용가능

    >> ERROR: (gcloud.compute.instances.create) Could not fetch resource:
    >> - Instances with guest accelerators do not support live migration.
  ```

4. 결국 나는 guest 이므로 accelerators (GPU) 사용 불가능. 하...

## 7. snapshot 과 image를 이용해서 vm-instance 복사 및 저장하기
- 참고 사이트 [https://geekflare.com/clone-google-cloud-vm/](https://geekflare.com/clone-google-cloud-vm/) 원래 알던 사이트는 아니다...

1. snapshot 만들기
2. Image 만들기
3. 프로젝트 구성원 추가하기
4. 다른 계정에서 인스턴스 만들기
5. 이미지 선택 -> 맞춤 이미지 -> 프로젝트 선택 -> 아까 만든 이미지 선택
6. 이미지 공유 성공!(공유는 불가능하고 migration은 가능하다)