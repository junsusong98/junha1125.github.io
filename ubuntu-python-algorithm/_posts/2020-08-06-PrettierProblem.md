---
layout: post
title: 【VScode】 Prettier 문제 해결을 위한 고찰
description: >
  Prettier 문제 해결을 위한 과정 기록
---

Prettier 문제 해결을 위한 과정 기록

### 문제점

- junha1125.github.io 폴더의 파일은 Prettier이 동작하지 않는 문제

  ![image](https://user-images.githubusercontent.com/46951365/91627900-cf283700-e9f5-11ea-81a9-a1708e539fce.png)

### 구글링을 통한 해결 과정

1. setting - format save on - 위의 사진과 같은 에러가 뜸.
2. 하지만 windows VScode, windows 다른 폴더에서 작업을 하면 Prettier 정상작동 md, py 파일 모두 정상 작동
3. 굳이 junha1125.github.io 이 폴더에서만 위와 같은 에러로 Formating 동작 안함
4. setting - Prettier : Prettier path 를 npm -global~을 실행해서 나오는 결과로 적어 넣으라고 해서 그렇게 함. 위의 에러는 안 뜨지만 동작하지 않음
   [이 사이트](https://github.com/prettier/prettier-vscode/issues/1066)의 조언
5. junha1125.github.io 폴더를 완전히 전체 삭제한 후, C 드라이브 위치말고 ~/user/sb020 내부 onedrive위치에 옮겨놓음. 그래도 동작 안함
6. 그러나 WSL에서는 모든 것이 잘 동작. 어느 폴더에서 어떤 파일형식으로 동작하든 잘 동작.

### 결론

1. 어떤 폴더든 어떤 파일이든 Pretter는 window에서도 WSL에서도 잘 동작한다.
2. 딱 하나! VScode - Open Folder - junha1125.github.io 로 열어서 작업하면 동작하지 않는다. (windows든 WSL이든)
3. 따라서 md파일을 수정하면서 prettier효과를 보고 싶다면, md 파일 하나만 VScode로 열어서 작업하라.
4. 또는 \_post 폴더 위에서 작업해도 Prettier 효과를 잘 볼 수 있다.

### 해결

- md파일을 수정하면서 prettier효과를 보고 싶다면, md 파일 하나만 VScode로 열어서 작업하라.
- 근데 에러는 뜬다. (동작 잘하는데 왜 똑같은 에러가 뜨는거지??????)
- 아하! 그렇다면 위의 링크 사이트대로 Prettier Path 수정하니까 에러도 안뜨고 동작도 잘 함(물론 Working Folder가 조금 꺼림직함...)

### 추가 조언

- Powershell, WSL, git Bash를 VS vode에서 동작하게 하는 방법.
  1. Default Shell 클릭 -> 원하는 Shell 선택
  2. VScode - Terminal - New Terminal
  3. 내가 아까 원했던 Shell이 나오는 것을 확인할 수 있다.

- 막상 Prettier이 있는 상태로 md파일을 수정하니 너무 불편하다. 그래서 Format Save 설정을 꺼버렸다. 

**문제 해결 완료.**
{:.lead}
