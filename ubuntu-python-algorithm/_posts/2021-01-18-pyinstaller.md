---
layout: post
title: 【Pyinstaller】Python file to EXE file (파이썬 파일 실행 파일로 만들기)
description: >
---

Python file to EXE file 파이썬 파일 실행 파일로 만들기 

참고 동영상 : [https://www.youtube.com/watch?v=UZX5kH72Yx4](https://www.youtube.com/watch?v=UZX5kH72Yx4)

```sh
$ pip install pyinstaller

$ pyinstaller --onefile -w <file_name.py>
```

1. 옵션

    --onefile : 깔끔하게 exe 파일 하나만 만들어 줌

    -w : console을 열지 않음. 만약 터미널 창의 결과가 보이거나 터미널 창에 값을 입력할 console이 보여야 한다면, -w 옵선 설정하지 말기

2. 실행 결과
    - 2개의 폴더 생성 build, dist
    - build는 지워버림 필요없음
    - dist에 .EXE 파일이 존재할 것임
    - 그거 그냥 실행하면 됨.

- **주의할점!! 의존성이 있는 경우** : EXE파일을 dist내부에서 실행하지 말고, (아마 "$ cd ../" 위치) 바로 아래 디렉토리 위치에 EXE파일을 옮겨놓고 파일 실행하기. 
    - from .ssd.modeling import detector
    - 이와 같은 문장을 python 파일에 적어놨다면, 파일의 위치가 일치해야 하므로.

3. **NSIS** 라는 프로그램 이용하기
    - 이 프로그램을 이용하면, 위와 같은 의존성 문제 고려 안해도 된다.
    - 위의 방법은 패키지 내부의 .py 파일 하나만을 exe파일로 변경하는 것이 었다. 
    - 패키지 전체를 하나의 installer 파일로 만들고 싶다면 이 프로그램을 이용해라.
    - <img src='https://user-images.githubusercontent.com/46951365/104916799-e2a06280-59d5-11eb-8c56-0890ba2d7fad.png' alt='drawing' width='400'/>
    - [Donwload link](https://nsis.sourceforge.io/Download)
    - 프로그램 사용방법은 맨 위 유투브 링크의, 6:30~8:30에 있음. 생각보다 매우 쉬우니 필요하면 해보기.
    - 

