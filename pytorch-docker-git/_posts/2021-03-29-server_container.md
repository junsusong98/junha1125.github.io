---
layout: post
title: 【docker】Windows10에서 원격서버의 docker container에 접속하기
---



-  **최종결과 화면**    
  ![image-20210329204418671](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora-rcv/image-20210329204418671.png?raw=tru)
-  꼭 reference 사이트에 들어가서 순서대로 따라할 것. 
-  사이트 내부의 내용을 모두 정리하지 않을 계획이다. 



# Windows10에서 원격서버의 docker container에 접속하기

## 1. Reference

1. [https://www.44bits.io/ko/post/wsl2-install-and-basic-usage](https://www.44bits.io/ko/post/wsl2-install-and-basic-usage)
2. [https://hwan-shell.tistory.com/182](https://hwan-shell.tistory.com/182)
3. [https://seokhyun2.tistory.com/42](https://seokhyun2.tistory.com/42)
4. [https://seokhyun2.tistory.com/48](https://seokhyun2.tistory.com/48)



## 2. WLS 및 docker desktop 설치

- 참고 : [https://www.44bits.io/ko/post/wsl2-install-and-basic-usage](https://www.44bits.io/ko/post/wsl2-install-and-basic-usage)
-  WLS2와 docker desktop을 설치한다. 이 과정이 꼭 필요한 과정인지는 모르겠다. vscode에서 `attach container`를 하면 `make sure docker deamon is running` 이라는 에러가 났다. 이 문제를 해결하기 위해서 docker desktop을 설치해야하는 듯 했다. 
- windows home을 사용하기 때문에 WLS2를 설치해야만 docker desktop을 설치할 수 있었다. 
- 사이트 순서를 정리하면 다음과 같다
  1. Windows Terminal 설치하기
  2. WSL 활성화를 위해서 PowerShell에서 두개의 명령어를 복붙 처주기
  3. WSL 설치하기 = MS store에서 Ubuntu 설치하기
  4. WSL1을 2로 업데이트 시켜주기 ([커널을 설치](https://docs.microsoft.com/ko-kr/windows/wsl/wsl2-kernel)해주고 `PowerShell에서 $ wsl --set-version/default` 처리 해주기 )
  5. Docker Desktop 설치하기 (windows home이지만, WSL2를 설치해놨기에 docker가 잘 설치가 되었다.)
  6. Docker 세팅 바꿔서 Ubuntu Terminal in window10에서도 docker 사용할 수 있게 만들기



## 3. VScode에서 SSH-remote 연결하기

- `Remote - Docker Containers` extension만 설치하면 안된다. 우선 SSH 연결이 필요하다

- 따라서 아래의 사이트를 참고 해서 다음의 과정을 진행해준다.

  - [https://seokhyun2.tistory.com/42](https://seokhyun2.tistory.com/42)
  - [https://hwan-shell.tistory.com/182](https://hwan-shell.tistory.com/182)
  - [https://data-newbie.tistory.com/688](https://data-newbie.tistory.com/688)
  - https://jmoon.co.kr/183

- 순서 정리

  1. window vscode에서 `Remote Development` extention 설치 (Remote 관련 extention 다 설치된다)

  2. Ubuntu Server 아래의 명령어 실행

     - ```sh
       $ sudo apt-get update
       $ sudo apt-get install ssh
       $ sudo apt-get install openssh-server
       $ sudo nano /etc/ssh/sshd_config # port 22 만 주석 풀어주기
       $ sudo service ssh status
       $ sudo service ssh start
       방화벽
     $ sudo ufw enable
       $ sudo ufw allow 22
     $ sudo ufw reload
       ```
  
  3. 윈도우 agent 세팅해주기

     - 관리자모드 CMD
   - `$ sc config ssh-agent start=auto` 
     - `$ net start ssh-agent` 
  
  4. 윈도우에서 ssh-keygen해서 ubuntu에 넣어주기 ([사이트](https://seokhyun2.tistory.com/42))
  
     - Powershell : `$ ssh-keygen -t rsa -b 4096`
     - Powershell : `$ Get-Content .\.ssh\id_rsa.pub`
   - 그러면 출력되는 긴~~ 결과물이 있을텐데, 그것을 복사해서 메일로든 뭐든 저장해두기
     - Ubuntu : `$ code /home/junha[=userName]/.ssh/authorized_keys`
   - Ubuntu : `$ chmod 644 /home/junha/.ssh/authorized_keys` : 이렇게 설정해주면 윈도우에서 ssh연결할때 우분투 비밀번호 안물어본다.
     - 여기에 그~대로 key 복붙해서 저장해두기 (아래 이미지 참조)
   - (에러 발생 및 문제 해결) 같은 아이피를 사용하는 우분투 케이스를 바꿔서 다시 연결하려니 안됐다. 이때 해결책 : 윈도우에서 known_hosts파일 삭제하기. `$ cd C:\Users\sb020\.ssh && rm ./known_hosts`
  
  5. 윈도우 VScode ssh config
  
     - VScode -> ctrl+shift+p == F1 -> `Remote-SSH: Connect to Host`
  
     - ```sh
     Host server1
       	HostName 143.283.153.11 # 꼭! ifconfig해서 ip확인
     	User junha
       	IdentityFile ~/.ssh/id_rsa
       ```

     - 위와 같이 config 파일저장해두기

     - 그리고 SSH VScode로 연결해보면 아주 잘된다.     
     ![image-20210329211438915](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora-rcv/image-20210329211438915.png?raw=tru)
  

  
## 4. Ubuntu Docker 설치 및 Container 실행
  
  1. [Docker 설치](https://docs.docker.com/engine/install/ubuntu/) : 기본 Docker 를 설치해줘야 NVIDA docker로 업그레이드 가능
  
  2. [우분투에 NVIDA Driver 설치](https://www.oofbird.me/55) 
  
   - `$ sudo ubuntu-drivers devices` 
     - `$ sudo ubuntu-drivers autoinstall`
   - `$ reboot`
     - 꼭 드라이버 설치하고, $ nvida-smi 하기 전에, reboot 꼭 해보기. 
     - `$ nvidia-smi`
  
  3. [NVIDIA-Docker 설치](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
  
     - ```sh
       # 정리
       1$ curl https://get.docker.com | sh \
         && sudo systemctl --now enable docker
       2$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
          && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
          && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
     3$ sudo apt-get update
       4$ sudo apt-get install -y nvidia-docker2
     5$ sudo systemctl restart docker
       6test-$ sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
       ```
  
  4. [ML workspace](https://github.com/ml-tooling/ml-workspace) Docker Image 다운로드(pull) 및 container Run
  
     - ```sh
       $ sudo docker run -d \
           -p 8080:8080 \
           --name "8080ML" \
           --runtime nvidia \
           --env NVIDIA_VISIBLE_DEVICES="all" \
           -v "${PWD}/docker_ws:/workspace" \
         --env AUTHENTICATE_VIA_JUPYTER="junha" \
           --shm-size 512m \
           --restart always \
           mltooling/ml-workspace:0.12.1
       ```
  
     - ```sh
       ### 설명 추가
       $ sudo docker run -d \ # background 실행
           -p 8080:8080 \ # 앞 포트는 내 우분투 포트, 뒤 포트는 컨테이너 내부 우분투 포트, 실험에 의하면 host-port/container-port이며, host-port만 바꿔주고 container-port는 8080으로 고정해서 사용하자
           --name "8080ML" \ # 맘대로 해라
           --runtime nvidia \ # 이거 꼭! 
           --env NVIDIA_VISIBLE_DEVICES="all" \ # 이것도 꼭 
           -v "${PWD}/docker_ws:/workspace" \ # 터미널실행PWD=="${PWD}"/docker_ws 폴더 만들어 놓기
         --env AUTHENTICATE_VIA_JUPYTER="junha" \ # 비밀번호
           --shm-size 512m \ # 일단 해놈. 안해도 될듯 
         --restart always \ # 알지? 
           mltooling/ml-workspace:0.12.1 # docker-hub에 버전 참고
     ```
  

  

  
  ## 5. Docker Container 연결하기

  1. 참고

     - [https://hwan-shell.tistory.com/182](https://hwan-shell.tistory.com/182)
     - [https://data-newbie.tistory.com/688](https://data-newbie.tistory.com/688)
  
  2. VScode -> ctrl+shift+p == F1 -> `Preferences: Open Settings (JSON)`
  
3. ```json
     {
       "docker.host": "ssh://junha@143.283.153.11"
     }
     ```
  
4. 이렇게 수정해두기. 위에서 junha로 하는 대신 우분투에서 아래의 작업 진행
  
5. ```sh
     $ sudo usermod -aG docker junha
   ```
  
  6. VScode -> ctrl+shift+p == F1 -> `Remote-Containers: Attach to Running Container`
  
  7. 존버타고 있으면, 서버에 실행되고 있는 container 목록이 뜬다. 원하는거 선택하고 Open 및 설정이 완료되기까지 시간이 좀 걸린다. 
  
  8. 그러면 최종 성공!    
     ![image-20210329211903648](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora-rcv/image-20210329211903648.png?raw=tru)





## 6. 완성!

- **Windows10**

![image-20210329213115925](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora-rcv/image-20210329213115925.png?raw=tru)

- **Ubuntu**

![image-20210329213208677](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora-rcv/image-20210329213208677.png?raw=tru)





## 7. VS code 환경설정

1. 아래와 같이 Python Interpreter 설정을 해야한다. 
2. ML-workspace자체에 conda가 설치되어 있고, base env는 `opt/conda/bin/python` 에 존재한다. interpreter에 마우스 길게 갖다 대고 있으면 path가 나온다. 나의 window conda path와 겹칠일은 없겠지만 그래도 조심하자.   
   ![image-20210329223237549](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210329223237549.png)

