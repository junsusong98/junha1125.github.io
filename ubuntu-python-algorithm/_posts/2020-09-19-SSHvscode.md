---
layout: post
title: 【GPU_Server】 Google Cloud 플랫폼 - VScode SSH, SSH 란?
description: > 
    VScode를 SSH를 사용하여 remote연결하기
---

VScode를 SSH를 사용하여 remote연결하기   

# PS
onedrive의 파일 wget으로 가져오기    
[stackoverflow 답변 링크](https://unix.stackexchange.com/questions/223734/how-to-download-files-and-folders-from-onedrive-using-wget)  
```
파일에 오른쪽 클릭하면 '임베드' 찾을 수 있음

<iframe src="https://onedrive.live.com/embed?cid=FBB2D8FB591497AB&resid=FBB2D8FB591497AB%2155221&authkey=AL6ELFsqIIuTp_s" width="98" height="120" frameborder="0" scrolling="no"></iframe>


https://onedrive.live.com/download?cid=FBB2D8FB591497AB&resid=FBB2D8FB591497AB%2155221&authkey=AL6ELFsqIIuTp_s

wget --no-check-certificate "https://onedrive.live.com/download?cid=FBB2D8FB591497AB&resid=FBB2D8FB591497AB%2155221&authkey=AL6ELFsqIIuTp_s"
```

# 0. Reference
1. 참고 동영상 [https://www.youtube.com/watch?v=7kum46SFIaY](https://www.youtube.com/watch?v=7kum46SFIaY)
2. 참고 이전 포스트 : [GCP의 instance에 Putty gen해서 생성한 key를 넣어주고 Putty로 SSH열기](https://junha1125.github.io/ubuntu-python-algorithm/2020-08-05-Google_cloud1/#3-%EB%94%A5%EB%9F%AC%EB%8B%9D%EC%9A%A9-%EA%B0%80%EC%83%81%ED%99%98%EA%B2%BD-%EA%B5%AC%EC%B6%95%ED%95%98%EA%B8%B0)
3. 참고 동영상 [인프런 SSH](https://www.inflearn.com/course/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%BB%B4%ED%93%A8%ED%84%B0%EB%B9%84%EC%A0%84-%EC%99%84%EB%B2%BD%EA%B0%80%EC%9D%B4%EB%93%9C/lecture/38573?tab=curriculum)
4. 1번 reference로 안돼서 [**이 블로그 글 이용**](https://amanokaze.github.io/blog/Connect-GCE-using-VS-Code/)   
    이 블로그 글에 대한 내용을 pdf로 저장해 두었다. 저장 공간은 다음과 같으니 나중에 필요하면 사용하기. [다운 링크](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2020-09-17/VS%20Code%EB%A5%BC%20%EC%82%AC%EC%9A%A9%ED%95%98%EC%97%AC%20Google%20Compute%20Engine%EC%97%90%20%EC%97%B0%EA%B2%B0%ED%95%98%EA%B8%B0.pdf)   
    
5. 혹시 모르니 나중에 이것도 참조 [https://evols-atirev.tistory.com/28](https://evols-atirev.tistory.com/28)
 
# 1. 설치 
1. SSH Gen 생성해서 vm instance에 집어넣어주기

    ![image](https://user-images.githubusercontent.com/46951365/93659936-0f268b00-fa85-11ea-85fd-082435682915.png)

2. VScode에 remote - SHH 설치하기 

    ![image](https://user-images.githubusercontent.com/46951365/93659918-f74f0700-fa84-11ea-96bd-29c8ccda0491.png)

3. SSH target 설정하기

    ![image](https://user-images.githubusercontent.com/46951365/93659959-5f055200-fa85-11ea-937d-68b004e8ad4e.png)

4. SSH + 하기

    ![image](https://user-images.githubusercontent.com/46951365/93660617-d8a03e80-fa8b-11ea-8432-68ca41e317f5.png)

5. config 저장 공간 설정

    ![image](https://user-images.githubusercontent.com/46951365/93660035-fb2f5900-fa85-11ea-9c78-e2e20402a987.png)

    맨 위에 있는 것을 선택헤서, SSH를 연결할 때마다 config가 삭제되지 않게 막아준다. 

6. Open 끝

    ![image](https://user-images.githubusercontent.com/46951365/93660631-f077c280-fa8b-11ea-89e1-a711e1f08bc2.png)

    ![image](https://user-images.githubusercontent.com/46951365/93660725-c96dc080-fa8c-11ea-8f92-e2166c5198b2.png)


# 2. 장점
1. 인터넷 속도 개 빠르다. 파이썬 패키지 다운로드 속도 미쳤다. 
2. CPU도 개좋다. 나의 컴퓨터 CPU를 전혀 사용하지 않는다.
3. 내 컴퓨터 메모리를 전혀 사용하지 않는다. WSL로 접속하면 Vmmeo로 컴퓨터 랩 1G를 잡아먹어서 짜증났는데, GCP를 사용하니 그런 일 없다.
4. 파일 보기 및 terminal 다루기가 편하다. 
5. Jupyter Notebook보다 더 많은 작업을 쉽게 할 수 있다. 
6. 파일을 그냥 끌어와서 업로드, 다운로드 할 수 있다. 개미쳤다.

# 3. 단점
1. 크레딧 다쓰면 진짜 내 돈나간다.
2. 다운로드 했던 많은 파일들 설정들이, Instacne 문제가 발생하면 한꺼번에 날아간다. 

# 4. 추신
1. 앞으로 WSL에 들어가서 굳이 작업하지 말자. WSL에서 작업할 이유가 전혀 없다. 
2. WSL을 전체 삭제해도 지금 될 것 같다. 하지만 일단 놔두자. 추가 환경설정 ~/.zshrc 파일이 나중에 도움이 될지도 모른다
2. zsh 설정한 것은 매우 좋지만, 굳이 더 이상 쓸 필요 없다. 만약 필요하면 다른 우분투 서버 환경을 zsh 설치해서 shell사용하면 된다. 


# 5. SSH 란?
- 강의 듣기 : [https://opentutorials.org/module/432/3738](https://opentutorials.org/module/432/3738)
- Client로 명령을 내리고 Server는 명령을 수행한다. 
- Linux에서는 Client가 이미 설치되어 있다. 
- 하지만 윈도우에서는 PuTTY로 SSH Client를 수행할 수 있다. 
- SSH server 설치 과정은 다음과 같다 ([참고 사이트1](https://technote.kr/318), [참고 사이트2](http://programmingskills.net/archives/315), [참고 사이트3](https://wlsvud84.tistory.com/12))
    1. $ sudo apt install openssh-server
    2. $ netstat -an|grep :22    
        여기서 22번 포트가 LISTEN 라면 openssh-server가 잘 동작중이라는 것.
    3. $ ifconfig 를 통해서 ip를 알아내고 putty에 적어서 실행.
    4. [putty 사용법](https://opentutorials.org/module/432/3740) 

- SSH Key - 비밀번호 보다 높은 수준의 보안이 필요할 때 사용.
    - [참고 사이트1](https://opentutorials.org/module/432/3742), [참고 사이트2](https://arsviator.blogspot.com/2015/04/ssh-ssh-key.html), [참고 사이트3 - putty Gen](http://taewan.kim/post/ssh_key/)
    - 


