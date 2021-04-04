---
layout: post
title: 【docker】How to use dockerfile & docker run in detail
---

-  공개된 Github 코드들 중에 Facebook, Microsoft, Google과 같은 대형 기업이 만든 패키지는 모두 Docker Installation을 제공한다. 즉 dockerfile이 제공되는데, 이것을 이용하는 방법에 대해서 간략히 공부해본다.
-  ([이전 Post 링크](https://junha1125.github.io/blog/pytorch-docker-git/2020-02-18-tstory_7/)) docker hub에서 Image를 가져올 때는 `pull` 명령어! 내가 Image를 배포하고 싶으면 `push` 명령어! 
-  미리 알아두어야할 핵심은, **dockerfile로 Image를 만드는 것이다. container 만드는 것은 docker run**



# 1. How to use dockerfile in order to make docker Image

## (0) Reference

1. [https://tech.osci.kr/2020/03/23/91695884/](https://tech.osci.kr/2020/03/23/91695884/)
2. [https://javacan.tistory.com/entry/docker-start-7-create-image-using-dockerfile](https://javacan.tistory.com/entry/docker-start-7-create-image-using-dockerfile)



## (1) reference summary

- Docker File이란 Docker Image를 만들기 위한 여러가지 명렁어의 집합이다. 

- 첫번째 reference에서는 아래의 일렬의 과정을 수행하기 위한 명령어들은 무엇이고, 이 과정이 자동으로 수행되게 만들기 위해서 dockerfile에는 어떤 내용이 들어가야하는지 설명한다. 

  1. ubuntu 설치
  2. 패키지 업데이트
  3. nginx 설치
  4. 경로설정 (필요한 경우)

- 최종적으로 dockerfile은 아래와 같이 만들어 질 수 있다.         

  ```sh
  FROM ubuntu:14.04 
  MAINTAINER Dongbin Na "kbseo@osci.kr" 
  RUN apt-get update
  RUN apt-get install -y nginx
  WORKDIR /etc/nginx 
  CMD ["nginx", "-g", "daemon off;"]
  EXPOSE 80 
  ```

- 이 내용을 한줄한줄 설명하면 다음과 같다.     

  ```sh
  1. FROM ubuntu:14.04 -> 기반으로 할 이미지를 가져옵니다. 여기에서는 ubuntu 14.04버전의 이미지를 가져오겠습니다.
  2. MAINTAINER Dongbin Na "kbseo@osci.kr" -> 작성자의 정보를 기입해줍니다.
  3. RUN apt-get update -> RUN이라는 명령어를 통하여 쉘 스크립트를 실행하여 줍니다.
  4. RUN apt-get install -y nginx -> 도커 빌드 중에는 키보드를 입력할 수 없기에 [-y] 옵션을 넣어줍니다.
  5. WORKDIR /etc/nginx -> cd와 같다. CMD 명령어가 실행 할 경로로 먼저 이동을 해 줍니다.
  6. CMD ["nginx", "-g", "daemon off;"] -> nginx를 백그라운드로 실행합니다
  7. EXPOSE 80 -> 80번 포트를 오픈하여 웹서버에 정상적으로 접근할 수 있게 합니다.
  
  필요 내용만 정리 (명령어, 명령어 역할, 예시)
  RUN	- 직접적으로 쉘 스크립트 내에서 실행 될 명령어 앞에 적어줍니다.	RUN <command>	RUN apt-get update
  2. CMD	- 도커가 실행될 때 실행할 명령어를 정의해줍니다.	CMD ["executable", "param", "param"]	CMD ["nginx", "-g", "daemon off;"]
  3. WORKDIR	- 이후 명령어가 작업할 디렉토리로 이동합니다	WORKDIR /path	WORKDIR /etc/nginx
  4. COPY	- 파일이나 디렉토리를 이미지로 복사합니다	COPY <src> <dst>	COPY . /usr/src/app
  5. ADD	- COPY와 비슷하게 복사를 위해 사용합니다	ADD <src> <dst>	ADD . /usr/src/app
  6. EXPOSE	- 공개 하고자 하는 포트를 지정해줍니다	EXPOSE <port>	EXPOSE 80
  ```

- 만든 dockerfile을 build 하기 `$ docker build --force-rm --tag mynginx:0.1 .` 

  - docker build가 실행되는 터미널 pwd에 있는 dockerfile을 자동으로 build해준다.
  - --force-rm : 중간중간에 생성되는 임시 container를 항상 remove 한다. (`--force-rm: Always remove intermediate containers`) 
  - --tag : 태그를 설정해줍니다. (docker-image-name : tag-name)



## 나의 사용 예시

- 실제 DETR docker Image를 build하기 위해 아래의 과정을 수행했다.    	

  ```sh
  $ cd ~/git-package/
  $ git clone https://github.com/facebookresearch/detr.git
  $ cd ./detr
  $ sudo docker build ./ # 이렇게 하면 아래처럼 대참사 발생
  $ sudo docker build --force-rm --tag ImageName:tag ./  #이렇게 사용하라. 
  # 처음에 ./ 를 안해줬더니 아래 같은 에러가 떴다.
  # "docker build" requires exactly 1 argument. Usage : docker build [Options] Pash 
  ```

- detr의 dockerfile을 보면 다음과 같다.    

  ```sh
  ROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime
  ENV DEBIAN_FRONTEND=noninteractive
  RUN apt-get update -qq && \
      apt-get install -y git vim libgtk2.0-dev && \
      rm -rf /var/cache/apk/*
  RUN pip --no-cache-dir install Cython
  COPY requirements.txt /workspace
  RUN pip --no-cache-dir install -r /workspace/requirements.txt
  ```

- 생각해보니까, 내가 굳이 Physical server (내 데스크탑)에 detr package 전체를 git clone할 필요가 없었다. 

- 어차피 dockerfile에 의해서 만들어지는 이미지안에 detr package가 들어갈 거다. 그리고 그 package내부의 requirement.txt가 pip install 될 것이다. 

- --tag 옵션으로 Image 이름 꼭 설정해주자... 안그러면 아래와 같은 대참사가 발생한다.    
  ![image](https://user-images.githubusercontent.com/46951365/113404779-af218180-93e3-11eb-8124-c9fb93615169.png?raw=tru)

- 나는 하나의 Image만 build하고 싶은데, 왜 많은 Image가 생성된걸까?   
  ![image-20210402185216777](https://user-images.githubusercontent.com/46951365/113405511-eb091680-93e4-11eb-9fa9-1a8215dfc359.png?raw=tru)

  - 우선 위의 대참사를 해결하기 위해서 --force-rm --tag 옵션을 넣어서 다시 수행했을때, 왠지는 모르겠지만 빠르게 Image가 build 되었다. 기존에 90f7 ca40 6296 4789 와 같은 Image가 존재하기 때문이다. 
  - dockerfile을 build하는 과정은 다음과 같다. 
    1. 맨 위에 명령어부터 차근차근 실행한다. 
    2. docker-hub에서 이미지를 다운받는다.
    3. 그 이미지로 container를 생성한다.
    4. 그 container에서 우분투 명령어를 RUN한다. 
    5. 그렇게 만들어진 container를 Image로 저장한다.  (--force-rm옵션에 의해서 위의 임시 container를 삭제한다.)
    6. 이 과정을 반복해서 많은 Image가 생성된다. 
    7. 최종적으로 만들어지는 Image는 `Successfully built 7f745326ad49` 에서 명시된 이미지이다. 





**지금까지 dockerfile을 build함으로써 docker Image는 만들었다!** 

**이 다음부터는 $ sudo docker run 의 문제이다.** 



# 2. docker run

- Reference: [docker run 커맨드 사용법](https://www.daleseo.com/docker-run/)
- [이전 Post 링크](https://junha1125.github.io/blog/pytorch-docker-git/2020-02-18-tstory_7/) : docker 기본 명령어 



## 나의 사용 예시

- 일단 아래처럼, ML-workspace 명령어 그대로 가져오면 안된다. 여기서 -it를 사용하지 않는 이유는 -it를 사용하지 않더라도 jupyter가 실행되며 terminal이 살아있기 때문이다. 

```sh
$ sudo docker run -d \
    -p 8888:8080 \
    --name "detr" \
    --gpus all\
    -v "${PWD}/docker_ws:/workspace" \
    --shm-size 512m \
    --restart always \ # exit상태가 되면 무조건 다시 start한다.
    7f745326ad49
```

- gpu사용하려면 이제 앞으로 --gpus 옵션만 넣으면 된다. 근데 이렇게 실행하면 이상하게 자동으로 exit가 되고 start를 수행하도 다시 exit가 된다. **이유** : -d 옵션을 사용하기만 하면 shell이 생성이 안되므로 자동으로 container가 죽어버린다

```sh
$ sudo docker run -d 7f74
```

- 이렇게 하니가 start하면 계속 살아있네? 

```sh
$ sudo docker run -it --gpus all  7f745326ad49
```

- 결론 -d -it 꼭 같이 사용하자...

```sh
$ sudo docker run -d -it --gpus all  7f745326ad49
```

- 다른 필요한 옵션도 추가 (**이거 쓰지말고 맨 아래에 새로 만들었으니 그거 써라**) 문제점 : -v에서 PWD가 들어가 있으니까, terminal PWD 생각안하고 container만들면 무조건 에러가 생기더라.

```sh
$ sudo docker run -d -it \
         --gpus all\
         --restart always\
         -p 8888:8080\	
         --name "detr" \
         -v "${PWD}/docker_ws:/workspace" \
         7f745326ad49
```

![image](https://user-images.githubusercontent.com/46951365/113412210-a4221d80-93f2-11eb-8bf5-846382e9ff1b.png?raw=tru)



## 3. Error 

1. 어제까지만 잘 실행되던 container가 VScode에서 오늘 안열린다. 

   - Error 내용 : /root/detr 이 존재하지 않는다.
   - 분석 : 내가 어제 git clone한 detr 폴더를 제거해버렸는데 그거 때문인가..?
   - 일단 아래 처럼 문제 해결

2. Dockerfile 이 실행되면서 마지막에 에러가 발생하고 이미지가 안 만들어진다. (어제까지는 잘만 되더만)

   - 무식하게 분석하지 말고 에러를 일고 확실히 분석해보자.       
     ![image-20210403180959468](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora-rcv/image-20210403180959468.png?raw=tru)

   - 가장 마지막 step 이후에 만들어진 Image를 이용해서 test를 해본다. 이럴때 `$ docker run --rm` 옵션을 사용하는 거구나   
     ![image-20210403181225402](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora-rcv/image-20210403181225402.png?raw=tru)

   - 그렇다면 submitit package는 나중에 따로 직접 설치하다 requirement.txt에서 submitit 일단 지워 놓자.

   - 오케이 일단! sumitit 패키지가 안들어간 이미지 생성 완성    
     ![image-20210403181805986](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora-rcv/image-20210403181805986.png?raw=tru)

   - 첫번째 에러가 해결되었다!!     
     ![image-20210403182238913](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora-rcv/image-20210403182238913.png?raw=tru)

     - 근데 처음에 /root/detr이 없다고 첫번쨰 에러가 발생했는데... 새로 생긴 container에서도 detr은 존재하지 않는데.... 어떻게 된것일까 모르겠다.

     - 가장 안정된 `docker run command`   

       - -v 옵션에서 조금이라도 문제가 있으면, 아에 -v 옵션이 적용이 안된다. 매우매우 주의할것

         ```sh
         sudo docker run -d -it      \
         --gpus all         \
         --restart always     \
         -p 8888:8080         \
         --name "detr"          \
         -v ~/docker/detr:/workspace   \
         detr:1
         ```

       - 앞으로 꼭!!! 이 과정으로 docekr 만들기

         1. `$ cd ~/docker ` 하고 거기서 원하는 package `$ git clone <packageA link>` 
         2. docker run의 -v옵션은 무조건  `~/docker/packageA:/workspace` 으로 설정
         3. 이렇게 하면 container의 `/workspace`에는 자동으로 packageA도 들어가 있고 아주 개꿀이다. 
         4. 참고로!! 혹시 모르니까 `packageA` 전체 내용은 삭제하지 말기.(첫번째 에러와 같은 문제가 다시 생길 수 있다.) 폴더를 만들고 싶다면 `packageA` 안에다가 폴더를 만들어서 거기서 작업하기
     
   - 아까 sumitit는 설치 못했으므로, 컨테이너에 직접 들어가서 패키지 설지    
     ![image-20210403183947407](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora-rcv/image-20210403183947407.png?raw=tru)

