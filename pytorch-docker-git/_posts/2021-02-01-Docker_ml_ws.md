---

layout: post
title: 【docker】container setting using ML-workspace 
---

- Github Link - [ml-tooling/ml-workspace](https://github.com/ml-tooling/ml-workspace)
- 과거에 공부했던 [docker 기본 정리 Post 1](https://junha1125.github.io/blog/pytorch-docker-git/2020-02-18-tstory_7/)
- 과거에 공부했던 [docker 기본 정리 Post 2](https://junha1125.github.io/blog/pytorch-docker-git/2020-02-21-tstory_8/)



## 1. [install docker](https://docs.docker.com/engine/install/ubuntu/) 



## 2. 필수 명령어 

- ```sh
  $ sudo docker image list
  $ sudo docker image rm <img_name>
  $ sudo docker container list -a 
  $ sudo docker container rm <container-ID>
  $ sudo docker container stop <container-ID>
  $ sudo docker container start <container-ID>
  ```



## 3. How to RUN ml-workspace

- ```sh
  docker run -d \
      -p 8080:8080 \  # port 설정 
      --name "ml-workspace" \
      -v "${PWD}:/workspace" \
      --env AUTHENTICATE_VIA_JUPYTER="mytoken" \
      --shm-size 512m \
      --restart always \
      mltooling/ml-workspace:0.12.1
  ```

- -p : [port 설정 설명](https://www.youtube.com/watch?v=pMY_wPih7R0&list=PLEOnZ6GeucBVj0V5JFQx_6XBbZrrynzMh&index=3)

- -v : "$pwd"/docker_ws:/workspace

  - [bind mount](https://docs.docker.com/storage/bind-mounts/)
  - 나의 컴퓨터 terminal의 현재 **"$pwd"**/docker_ws  (나같은 경우 **/home/junha**/docker_ws)
  - container의 new ubuntu root에 /workspace라는 폴더 생성 후

- -env AUTHENTICATE_VIA_JUPYTER="내가 설정하고 싶은 비밀번호"

- -d : background container start



## 4. Docker container와 VScode 연동하기

1. 첫번째 VScode 사용하는 방법
   - [ml-workspace에서 알려주는 vscode](https://github.com/ml-tooling/ml-workspace#visual-studio-code)
   - vscode 서버를 사용한 실행.
   - 원하는 폴더를 체크박스_체크 를 해주면, vscode옵션을 선택할 수 있다. 
   - 이렇게 하면 EXTENSIONS 를 나의 컴퓨터 그대로 사용하기는 어려우므로 VScode docker 연동법을 공부해보자.
2. 컴퓨터 VScode의 docker SSH 를 사용하서 연결하기.
   - Remote-container EXTENSION을 설치해준다. 
   - ctrl+shift+p -> Remote-Containers: Attach to Running Container
     - 자동으로 현재 container 목록이 보인다. 원하는거 고르면 remote vscode 새창 뜬다
     - 아주 잘 됨.
   - 굳이 IP설정을 해주지 않더라도 잘 연결된다. ([GCP에서의 SSH 설정하기](https://junha1125.github.io/blog/ubuntu-python-algorithm/2020-09-19-SSHvscode/) 이것과는 달리..)
   - 만약 문제가 생기면 config 파일 여는 방법
   - ctrl+shift+p -> Remote-Containers: Open Attached Container Configuration File

















