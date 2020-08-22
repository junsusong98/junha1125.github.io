---
layout: post
title: (Ubuntu) 노마드 코더의 윈도우 10 개발환경 구축
# description: > 
---

참고 사이트 : 
[노마드 코더의 윈도우 10 개발환경 구축](https://nomadcoders.co/windows-setup-for-developers/)  
## 1. Setup
1. windows Update  
windows 10 - 2004 버전까지 업데이트가 되어있어야 WSL를 사용가능하다.   
2. VScode  
vscode를 설치하고 다양한 extension을 설치하면 좋다.   
예를 들어서 나는 material Theme, material theme icons, prettier등이 좋았다.  
3. Chocolatey  
우리가 우분투에서 프로그램을 설치하려면   
$ sudo apt-get install kolourpaint4  
를 하면 쉽게 설치할 수 있었다. 윈도우에서도 이렇게 할수 있게 해주는게 Chocolatey이다. 설치방법은 간단하고, 이제 windows powershell에서 [Find packages](https://chocolatey.org/packages)에서 알려주는 명령어를 그냥 복사, 붙여넣기하면 프로그램 설치를 매우 쉽게 할 수 있다. 그럼에도 불구하고 이것보다 Linux환경을 사용하는 것을 더욱 추천한다. 
4. windows terminal  
MS store에서 다운받을 수 있는 터미널. 그리거 [여기 WSL](https://docs.microsoft.com/ko-kr/windows/wsl/install-win10)를 파워셀에 써넣어서 리눅스 계열 OS(Ubunutu) 설치할 수 있게 해준다. 그리고 MS store에 들어가서 Ubuntu를 설치해준다.(다시시작 반복 할 것)   
설치 후 바로 위의 사이트의 '~옵션 구성 요소 사용', '~기본 버전 설정', '~WSL 2로 설정'등을 그대로 수행한다. 

## 2. Terminal Customization  
1. $ sudo apt install zsh
2. [사이트](https://github.com/ohmyzsh/ohmyzsh)의 bash install을 이용해 우분투에 설치. curl, wget 이용한 설치든 상관없음. 
3. Oh my zsh 설치완료.
4. 테마 변경하기  
[이 사이트](https://terminalsplash.com/)를 이용해서 새로운 schems를 만들어주고 colorScheme을 변경해주면 좋다. 
5. Powerlevel10k  
[이 사이트](https://github.com/romkatv/powerlevel10k)로 좀 더 좋은 테마로 변경. 터미널을 다시 열면 많은 설정이 뜬다. 이때, 나는 추가로 위의 사이트 중간 부분에 존재하는 font 'MesloLGS NG'를 다운받고 윈도위 글꼴 설정에 넣어주었다. 그랬더니 모든 설정을 순조롭게 할 수 있었다. 그리고 신기하게 언제부터인가 터미널에서 핀치줌을 할 수 있다.(개꿀^^) 뭘 설치해서 그런지는 모르겠다. 
6. vscode 터미널 모양 바꿔주기  
setting -> Terminal › Integrated › Shell: Windows -> edit json -> "terminal.integrated.shell.windows": "c:\\Windows\\System32\\wsl.exe"  
7. ls color  

- 전체적인 과정을 하고 느낀점   
뭔가 혼자 찾으면 오래 걸릴 것을 순식간에 해버려서... 감당이 안된다. 이 전반적인 원리를 알지 못해서 조금 아쉽지만, 내가 필요한 건 이 일렬의 과정의 원리를 아는 것이 아니라 그냥 사용할 수 있을 정도로만 이렇게 설정할 수 있기만 하면되니까, 걱정하지말고 그냥 잘 사용하자. 이제는 우분투와 윈도우가 어떻게 연결되어 있는지 알아볼 차례이다.   
- powerlevel10k 환경설정을 처음부터 다시 하고 싶다면, $ p10k configure 만 치면 된다.
- **주의 할 점!!** 우분투에 ~/.zshrc 파일을 몇번 수정해 왔다. oh my zsh를 설치할 때 부터.. 그래서 지금 설치한 우분투 18.04를 삭제하고 다 깔면 지금까지의 일렬의 과정을 다시 해야한다. '## Terminal customization'과정을 처음주터 다시 하면 된다. 

## 3. Installing Everything
1. 우분투와 윈도우와의 관계  
$ cd /mnt(mount)/c(c드라이브) -> 우분투와 윈도우를 연결해주는 부분    
$ ls /mnt/c/Users/sb020 -> 결국에 여기가 나의 Document들이 있는 부분   
    - 여기서 내가 torch, vi 등으로 파일을 만들고 수정해도 가능하다. 즉 우분투 상에서 윈도우에 접근해서 뭐든 할 수 있는 것이다.   
    - 대부분의 WSL사용자들은 우분투 공간에서 윈도우 파일을 만들고 수정하지 않는다. 일반적인 우분투 사용자처럼 /home/junha/~ 에 파일이나 프로젝트를 넣어두고 다룬다. **하지만!!** 이러한 방법은 WSL에 문제가 생기거나 Ubuntu-18.04에 문제가 생겨서 지우게 된다면 발생하는 문제를 고스란히 감안해야한다. 따라서 노마드 쌤이 추천하길, **우분투 경로 말고 /mnt/c 위의 윈도우 경로에, 프로젝트 파일 등은 저장하고 다루는 것이 좋다.** 
    - 리눅스 콘솔에서 윈도우에 있는 파일을 건드릴 수 있다. 하지만 윈도우에서 리눅스 파일(/home/junha 맞나..? 잘 모르곘다. )을 건드린다면 그건 좋은 방법이 아니다. 문제가 발생할 수도 있다. 
    - conda에 대한 나의 생각 : zsh의 상태를 보면 conda를 쳐도 읽지를 못한다. 이 말은 conda는 윈도우의 powerShell이나 cmd에서만 동장한다. 따라서 우분투에 들어가서 항상 내가 아나콘다부터 설치한 듯이, 아나콘다를 다시 설치해야한다.^^
2. 우분투에 프로그램 설치하기  
