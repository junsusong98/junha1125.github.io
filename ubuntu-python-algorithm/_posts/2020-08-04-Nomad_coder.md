---
layout: post
title: 【Ubuntu】 노마드 코더의 윈도우 10 개발환경 구축
# description: > 
---


참고 사이트 : 
[노마드 코더의 윈도우 10 개발환경 구축](https://nomadcoders.co/windows-setup-for-developers/)    
이 과정을 메모리가 좀 더 높은 컴퓨터에 하고 싶은데... 지금 새로운 것을 사서 하기도 그렇고 일단 삼성 노트북 팬s에서 잘해보고 나중에 컴퓨터 새로 사면 또 다시 설치 잘 해보자^^  

 ## [Final setting Image]
 ![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2020-08-24/KakaoTalk_20200823_101100388.png?raw=true)

# 1. Setup
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

# 2. Terminal Customization  
1. $ sudo apt install zsh
2. [사이트](https://github.com/ohmyzsh/ohmyzsh)의 bash install을 이용해 우분투에 설치. curl, wget 이용한 설치든 상관없음. 
3. Oh my zsh 설치완료.
4. 터미널 테마 변경하기  
[이 사이트](https://terminalsplash.com/)를 이용해서 새로운 schems를 만들어주고 colorScheme을 변경해주면 좋다. 
    ```sh
        
    // To view the default settings, hold "alt" while clicking on the "Settings" button.
    // For documentation on these settings, see: https://aka.ms/terminal-documentation

    {
        "$schema": "https://aka.ms/terminal-profiles-schema",

        "defaultProfile": "{c6eaf9f4-32a7-5fdc-b5cf-066e8a4b1e40}",

        "profiles":
        {
            "defaults":
            {
                    "fontFace" : "MesloLGS NF"
            },
            "list":
            [
            
                {
                    // Make changes here to the powershell.exe profile
                    "guid": "{61c54bbd-c2c6-5271-96e7-009a87ff44bf}",
                    "name": "Windows PowerShell",
                    "commandline": "powershell.exe",
                    "hidden": false,
                    "colorScheme" : "Monokai Night"
                },
                {
                    // Make changes here to the cmd.exe profile
                    "guid": "{0caa0dad-35be-5f56-a8ff-afceeeaa6101}",
                    "name": "cmd",
                    "commandline": "cmd.exe",
                    "hidden": true
                },
                {
                    "guid": "{b453ae62-4e3d-5e58-b989-0a998ec441b8}",
                    "hidden": true,
                    "name": "Azure Cloud Shell",
                    "source": "Windows.Terminal.Azure"
                },
                {
                    "guid": "{c6eaf9f4-32a7-5fdc-b5cf-066e8a4b1e40}",
                    "hidden": false,
                    "name": "Ubuntu-18.04",
                    "source": "Windows.Terminal.Wsl",
                    "colorScheme" : "VSCode Theme for Windows Terminal"
                }

        
        
            ]
        },

        // Add custom color schemes to this array
        "schemes": [
            {
                "name" : "Monokai Night",
                "background" : "#1f1f1f",
                "foreground" : "#f8f8f8",
                "black" : "#1f1f1f",
                "blue" : "#6699df",
                "cyan" : "#e69f66",
                "green" : "#a6e22e",
                "purple" : "#ae81ff",
                "red" : "#f92672",
                "white" : "#f8f8f2",
                "yellow" : "#e6db74",
                "brightBlack" : "#75715e",
                "brightBlue" : "#66d9ef",
                "brightCyan" : "#e69f66",
                "brightGreen" : "#a6e22e",
                "brightPurple" : "#ae81ff",
                "brightRed" : "#f92672",
                "brightWhite" : "#f8f8f2",
                "brightYellow" : "#e6db74"
            },
            {
                "name" : "VSCode Theme for Windows Terminal",
                "background" : "#232323",
                "black" : "#000000",
                "blue" : "#579BD5",
                "brightBlack" : "#797979",
                "brightBlue" : "#9BDBFE",
                "brightCyan" : "#2BC4E2",
                "brightGreen" : "#1AD69C",
                "brightPurple" : "#DF89DD",
                "brightRed" : "#F6645D",
                "brightWhite" : "#EAEAEA",
                "brightYellow" : "#F6F353",
                "cyan" : "#00B6D6",
                "foreground" : "#D3D3D3",
                "green" : "#3FC48A",
                "purple" : "#CA5BC8",
                "red" : "#D8473F",
                "white" : "#EAEAEA",
                "yellow" : "#D7BA7D"
            }

        ],

        // Add any keybinding overrides to this array.
        // To unbind a default keybinding, set the command to "unbound"
        "keybindings": []
    }
    ```

5. 명령어 라인 테마 변경하기 - Powerlevel10k  
[이 사이트](https://github.com/romkatv/powerlevel10k#oh-my-zsh)로 좀 더 좋은 테마로 변경. 터미널을 다시 열면 많은 설정이 뜬다.   
잠시 발생하는 에러를 헤결하기 위해, 나는 추가로 위의 사이트 중간 부분에 존재하는 [font 'MesloLGS NG'를 다운](https://github.com/romkatv/powerlevel10k#manual-font-installation)받고 윈도위 글꼴 설정에 넣어주었다. 그랬더니 모든 설정을 순조롭게 할 수 있었다. 그리고 신기하게 언제부터인가 터미널에서 핀치줌을 할 수 있다.(개꿀^^) 뭘 설치해서 그런지는 모르겠다. 
6. vscode 터미널 모양 바꿔주기  
setting -> Terminal › Integrated › Shell: Windows -> edit json -> "terminal.integrated.shell.windows": "c:\\Windows\\System32\\wsl.exe"  
    ![image](https://user-images.githubusercontent.com/46951365/92989121-ab441580-f50c-11ea-9f1d-fec982d693b9.png)
7. [ls color](https://qastack.kr/ubuntu/466198/how-do-i-change-the-color-for-directories-with-ls-in-the-console) : code ~/.zshrc  

- 추가 메모
    - 뭔가 혼자 찾으면 오래 걸릴 것을 순식간에 해버려서... 감당이 안된다. 이 전반적인 원리를 알지 못해서 조금 아쉽지만, 내가 필요한 건 이 일렬의 과정의 원리를 아는 것이 아니라 그냥 사용할 수 있을 정도로만 이렇게 설정할 수 있기만 하면되니까, 걱정하지말고 그냥 잘 사용하자. 이제는 우분투와 윈도우가 어떻게 연결되어 있는지 알아볼 차례이다.   
    - powerlevel10k 환경설정을 처음부터 다시 하고 싶다면, $ p10k configure 만 치면 된다.
    - **주의 할 점!!** 우분투에 ~/.zshrc 파일을 몇번 수정해 왔다. oh my zsh를 설치할 때 부터.. 그래서 지금 설치한 우분투 18.04를 삭제하고 다 깔면 지금까지의 일렬의 과정을 다시 해야한다. '## Terminal customization'과정을 처음주터 다시 하면 된다. 

# 3. Installing Everything
1. 우분투와 윈도우와의 관계  
$ cd /mnt(mount)/c(c드라이브) -> 우분투와 윈도우를 연결해주는 부분    
$ ls /mnt/c/Users/sb020 -> 결국에 여기가 나의 Document들이 있는 부분   
    - 여기서 내가 torch, vi 등으로 파일을 만들고 수정해도 가능하다. 즉 우분투 상에서 윈도우에 접근해서 뭐든 할 수 있는 것이다.   
    - 대부분의 WSL사용자들은 우분투 공간에서 윈도우 파일을 만들고 수정하지 않는다. 일반적인 우분투 사용자처럼 /home/junha/~ 에 파일이나 프로젝트를 넣어두고 다룬다. **하지만!!** 이러한 방법은 WSL에 문제가 생기거나 Ubuntu-18.04에 문제가 생겨서 지우게 된다면 발생하는 문제를 고스란히 감안해야한다. 따라서 노마드 쌤이 추천하길, **우분투 경로 말고 /mnt/c 위의 윈도우 경로에, 프로젝트 파일 등은 저장하고 다루는 것이 좋다.** 
    - 리눅스 콘솔에서 윈도우에 있는 파일을 건드릴 수 있다. 하지만 윈도우에서 리눅스 파일(/home/junha 맞나..? 잘 모르곘다. )을 건드린다면 그건 좋은 방법이 아니다. 문제가 발생할 수도 있다. 
    - conda에 대한 나의 생각 : zsh의 상태를 보면 conda를 쳐도 읽지를 못한다. 이 말은 conda는 윈도우의 powerShell이나 cmd에서만 동장한다. 따라서 우분투에 들어가서 항상 내가 아나콘다부터 설치한 듯이, 아나콘다를 다시 설치해야한다.^^
2. bashrc NO!. zshrc Yes!.
    - 강의에서 nodejs를 설치했다. 나는 그동안 anaconda를 설치했다. /home/junha/anaconda에 설치가 되었다. /mnt/c/Users/sb020/anaconda와는 완전히 다른 것이다. 즉 내가 설치한 Ubunutu 18.04에 ubuntu를 설치한 것이고, c드라이브와 영향은 없는 것으로 추측할 수 있다. /ageron/handson-ml/blob/master/requirements.txt) 딥러닝에 유용한 package requirements를 다운받을 수 있었다. (python == 3.6 version이어야 requiremtnet.txt가 잘 작동)  
    - 이렇게 처음 설치하면 딥러닝 환경 설정으로 아주 편하다. conda를 사용해서 설정하는게 package 버전 관리에 매우 유용하다. pip을 사용해도 되지만 같이 쓰면 항상 문제가 발생하더라...  
        ```sh
        $ conda install Numpy Scipy Scikit-learn Theano TensorFlow Keras PyTorch Pandas Matplotlib      
        $ conda install -c conda-forge opencv  
        $ conda install -c pytorch torchvision
        ```  
    - 엄청난 것을 깨달았다. 왜 ~/.bashrc 에 conda에 관한 아무런 내용이 없지? 라고 생각했다.  왜냐하면 나는 지금 zsh shell을 사용하고 있기 때문이다. 따라서 ~/.zshrc 에 들어가면 conda에 대한 설정이 있었다. 
    - vi를 사용해서 파일을 수정할 필요가 이제 없다. $ vi ~/.zshrc 하지말고 $ code ~/.zshrc를 하면 매우 쉽다. (vscode 자동실행) 여기에 들어가서 alias를 이용해서 단축어를 만들어놨다.  
        ```
        alias python=python3.8  
        alias win="cd /mnt/c/Users/sb020"  
        alias acttor="conda activate torch"  
        ```

3. WSL ubuntu VScode  
    - ![image](https://user-images.githubusercontent.com/46951365/90973167-5c810c80-e55a-11ea-8ba4-55fec2aeca32.png)
    - 위의 사진에서 보이는 것처럼, 2가지의 운영체제에서 하나의 vscode를 사용하고 있다. 따라서 Extentions도 여러가지 다시 설치해줘야했다. 또한 VScode 맨아래 왼쪽에 WSL과 Local VScode로 이동할 수 있는 버튼이 있었다. 
    - prettier를 사용하면 코드를 save하면 코드를 이쁘게 다 재배열해준다. vscode에 가장 필요한 extentions라고 하는데 진짜인 것 같다. WSL setting에 들어가서 'editer format on save'설정을 해줘야한다. 윈도우, 우분투 vscode Setting은 완전히 다르다. 따라서 윈도우도 같은 설정을 해줬다. 
    - 하지만... 아래와 같이 이와 같은 오류가 떴다. "Failed to load module. If you have prettier or plugins referenced in package.json, ensure you have run `npm install` Attempted to load prettier from c:\projects\junha1125.github.io" 
    - 그래서 npm 설치하기 위해서 인터넷에서 찾아보니 nodejs를 다운받으라고 해서 [choco를 통해](https://chocolatey.org/packages/nodejs-lts#install) 빠르게 다운받았다. 그래도 안됐다. 
    - 아래의 사진과 같이 setting도 2가지 환경에서 서로 다르게 셋팅할 수 있으니 주의할 것.   
    ![image](https://user-images.githubusercontent.com/46951365/90973222-f779e680-e55a-11ea-8417-8d6e70ed3c8f.png)
    - **주의** 더이상 vi쓰지마라. $ code \<pathName or fileName or directoryName\>  

4. VScode 사용 주의사항
    - interpretter 설정에 주의하기  
    ![image](https://user-images.githubusercontent.com/46951365/93431783-99dd7d80-f8ff-11ea-861d-82edc7071956.png)

