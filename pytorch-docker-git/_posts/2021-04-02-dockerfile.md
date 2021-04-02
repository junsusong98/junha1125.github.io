---
layout: post
title: 【docker】How to use dockerfile
---

-  공개된 Github 코드들 중에 Facebook, Microsoft, Google과 같은 대형 기업이 만든 패키지는 모두 Docker Installation을 제공한다. 즉 dockerfile이 제공되는데, 이것을 이용하는 방법에 대해서 간략히 공부해본다.
-  ([이전 Post 링크](https://junha1125.github.io/blog/pytorch-docker-git/2020-02-18-tstory_7/)) docker hub에서 Image를 가져올 때는 `pull` 명령어! 내가 Image를 배포하고 싶으면 `push` 명령어! 



# How to use dockerfile

## 0. Reference

1. [https://tech.osci.kr/2020/03/23/91695884/](https://tech.osci.kr/2020/03/23/91695884/)
2. [https://javacan.tistory.com/entry/docker-start-7-create-image-using-dockerfile](https://javacan.tistory.com/entry/docker-start-7-create-image-using-dockerfile)



## 1. reference summary

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
  - --force-rm : 기존에 존재하는 image를 삭제합니다.
  - --tag : 태그를 설정해줍니다. (docker-image-name : tag-name)



## 나의 사용 예시

- 실제 DETR docker Image를 build하기 위해 아래의 과정을 수행했다.    	

  ```sh
  $ cd ~/git-package/
  $ git clone https://github.com/facebookresearch/detr.git
  $ cd ./detr
  $ sudo docker build ./
  # 처음에 ./ 를 안해줬더니 아래 같은 에가 떴다.
  # "docker build" requires exactly 1 argument. Usage : docker build [Options] Pash 
  ```

- detr의 dockerfile을 보면 다음과 같다.    
  ```









## 2. reference summary

