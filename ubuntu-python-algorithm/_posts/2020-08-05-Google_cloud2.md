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
    $ cd envs/tf113/lib/python3.6   -> python3.6 설치
    $ cd ./site-packages  -> 기본 package들이 설치되어 있다. 
    $ cd ~/DLCV/data/util/
    $ chmod +x *.sh
    $ ./install_tf113.sh  (중간에 에러나오는거 무시 - 버전 맞춰주는 것)

    $ ps -ef | grep jupyter 
    - jupyter lab 말고 /계정ID/anaconda3 jupyter PID 
    $ kill -9 PID
    $ conda activate tf113
    $ cd ~/ && ./start_jn.sh         
    - jupyter의 목록에 conda 라는 목록이 추가된 것을 확인할 수 있다. 
    ```

- 새로운 conda env를 사용해서 jupyter를 열고 싶으면 다음을 꼭 설치해야 한다.

    ```
    # DLCV/blob/master/data/util/install_tf113.sh 참조
    $ conda install -c conda-forge pycocotools -y
    $ conda install nb_conda -y

    # for  multi jupyter notebook kernal
    $ pip install ipykernel
    $ python -m ipykernel install --user --name <Conda Eev Name>
    ```



- install_tf113.sh 내부에 있는 코드 중 $ conda install [nb_conda](https://github.com/Anaconda-Platform/nb_conda) -y 명령어를 통해서 콘다 환경들에게 별도로 jupyter를 따로 실행할 수 있게 해준다.   
- 여러개의 conda 가상환경에서 Jupyter를 열 때. kernel을 골라달라고 창이 뜰 수도 있다. 이때 tf113을 잘 골라주면 된다. 자주 나오니까 각 버전에 맞는 kernel 환경을 선택해 주면 된다. 선택할 kernel 이름이 있는 이유는 install.sh부분에 '$ python -m ipykernel install --user --name tf113'이 있기 때문에 이름이 있는거다. env여서 이름이 뜨는게 아니라 커널 설정을 해줬기 때문에 이름이 뜨는 거다. (Jypyter에서 Python [conda env:tf113] 와 같은 노투븍 커널 셋팅을 할 수 있도록!)

![image](https://user-images.githubusercontent.com/46951365/91634184-71154700-ea29-11ea-84a9-a9a25be503c6.png)

- retinaNet에서 tf115를 사용하니까, 그때는 원래 띄어진 Jypyter를 죽이고 activate tf115 를 한 후 start_jn.sh를 실행한다. 



## 2. 구글 클라우드 $300 무료 크레딧 효과적 사용법
1. 결재 - 개요에 자주 들어가기
2. 결제 - 예산 및 알림 - 알림 설정하기
3. 결제 - 거래 - 자주 들어가서 확인하기
    - 300 GB storage PD capacity 한달에 14,000원 계속 나간다. 
    - CPU core, Static ip(고정 ip) charging 등 돈이 추가적으로 많이 들어간다.
4. GPU 서버를 항상 내리는 습관을 가지도록 하자. (VM instance 중지 및 정지)
5. GPU 사용량, Setup 하는 자세를 배우는 것도 매우 필요하다. 
6. 구글 클라우드 서비스 모두 삭제하는 방법
    - instance 중지 - 삭제 - 서버 및 storage 비용은 안나간다 
    - VPC network - 외부 IP 주소 - 고정 주소 해제 (이것도 매달 8천원 나간다.. )
    - Strage - 브라우저 - object storage 삭제
    - 프로젝트 설정 - 종료 (모든 결제 및 트래픽 전달 중지 된다)
7   결제 자주 확인하기  
    <img src="https://user-images.githubusercontent.com/46951365/91517045-ec8ecf80-e927-11ea-951e-2e04235235de.png" alt="image" style="zoom:67%;" />


## 3. Cloud 사용시 주의사항 및 Object Storage 설정
1. 오류 해결 방법 
    - GPU resources 부족 = 기다려라. 하루정도 기다리면 문제 해결 가능
    - 심각한 문제가 발생하면 VM instance 지우고 다시 설치 및 Setup하는 자세 및 습관을 들여라.
2. Object Storage 설정하기
    - 작업을 편하기 하기 위해서 설정해두면 편리하다. 
    - storage - 브라우저 - 버킷 생성 - 단일 region 및 설정 - default 설정 - 저장
    - 접근 인증을 받아야 한다. - server에서 object storage 에 붙기위해서 몇가지 설정이 필요.  

        ```sh
        gsutil : google cloud와 연동되어 쉽게 이용할 수 있는 터미널 명령어
        $ gsutil ls gs://my_budcker_Name
        - 아직 권한이 없는 것을 확인할 수 있다. 
        ```

    - object storage 세부정보 - 권한 - 구성원추가 - 계정이메일 추가, 역할 Storage 저장소 관리자, 저장 - 이제 storage 에 등록을 했다. 하지만 server 인증을 받아야 한다.  

        ```sh
        $ gcloud auth login
        - Yes
        - link copy
        - 웹 도메인에 paste - 로그인 및 코드 복사
        - SSH에서 Enter verification code
        $ gsutil ls gs://my_budcker_Name    -> 이제 여기 접근 가능
        $ qsutil cp <file Name> gs://my_budcker_Name    -> 이런식으로 bucket(object storage)에 접근 가능
        ```

    - winSCP를 사용해서 Putty와 연동 및 파일 업로드 가능
        - winSCP 다운로드 및 설치
        - 로그인 - 도구 - 가져오기 - Putty 원하는 환경 선택 
        - 파일 업로드가 매우 쉽게 가능
        - <img src="https://user-images.githubusercontent.com/46951365/91518024-32e52e00-e92a-11ea-9b29-47ecd02d2a95.png" alt="image" style="zoom:67%;" />


## 4. Colab을 이용한 실습 환경 구축하기
1. 런타임 유형 - GPU 설정 주의
2. tensorflow 1.13에서 동작이 안되므로, tensorflow 1.15 설치하기.
3. % cd, ! terminal command 와 같이 %와 !를 적절히 사용해라. 
4. google drive mount 해서 사용하기    
    ![image](https://user-images.githubusercontent.com/46951365/91525462-86607780-e93c-11ea-8d3a-3fab71096ed0.png)



