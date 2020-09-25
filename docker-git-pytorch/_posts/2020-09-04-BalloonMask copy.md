---
layout: post
title: 【Keras】Keras기반 Mask-RCNN - Balloon 데이터셋, Python 추가 공부
description: >
    Keras 기반 Mask-RCNN를 이용해 Balloon 데이터셋을 학습시키고 추론해보자.
---
Keras기반 Mask-RCNN - Balloon 데이터셋  

# 1. 앞으로의 과정 핵심 요약

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/93461157-5135bc80-f91f-11ea-9d82-fec2a49b5286.png' alt='drawing' width='500'/></p>

1. [/Mask_RCNN/tree/master/samples/balloon](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon) 에 있는 내용들을 활용하자. 

2. Matterport Mask RCNN Train 프로세스 및 앞으로의 과정 핵심 정리  
    <p align="center"><img src='https://user-images.githubusercontent.com/46951365/93462114-c35ad100-f920-11ea-88e6-8249dac2ffab.png' alt='drawing' width='700'/></p>  
    <p align="center"><img src='https://user-images.githubusercontent.com/46951365/93462130-c9e94880-f920-11ea-94d2-1dbe199288e5.png' alt='drawing' width='700'/></p>  

# 2. 스스로 공부해보기
- **강의를 보니, 큰 그림은 잡아주신다. 하지만 진짜 공부는 내가 해야한다. 코드를 하나하나 파는 시간을 가져보자. 데이터 전처리 작업으로써 언제 어디서 나중에 다시 사용할 능력일지 모르니, 직접 공부하는게 낫겠다.** 
    - 코드 보는 순서 기록
        1. /mask_rcnn/Balloon_데이터세트_학습및_Segmentation.ipynb
        2. /sample/balloon/balloon.py
        3. /mrcnn/utils.py

- Python 새롭게 안 사실 정리

    1. super() : [참고 사이트](https://rednooby.tistory.com/56)  
        - \_\_init\_\_ 나 다른 맴버 함수를 포함해서, 자식 클래스에서 아무런 def을 하지 않으면 고대로~ 부모 클래스의 내용이 상속된다. 자식 클래스에서도 함수든 변수든 모두 사용 가능하다.   
        - 하지만 문제가 언제 발생하냐면, def 하는 순간 발생한다. 만약 def \_\_init\_\_(self, ..): 하는 순간, 오버라이딩이 되어 원래 부모 클래스의 내용은 전부 사라지게 된다. 이럴 떄, 사용하는게 super이다.   
        - 대신 클래스 변수를 사용하는 공간에는 super를 사용하지 않아도 상속한 부모 클래스의 내용이 전부 알아서 들어간다. 

    2. VS code - [새롭게 파일을 열 떄 강제로 새로운 탭으로 나오게 하는 방법](https://stackoverflow.com/questions/38713405/open-files-always-in-a-new-tab) : setting -> workbench.editor.enablePreview" -> false 체크

    3. jupyter Notebook font size 바꾸기 : setting -> Editor: Font Size Controls the font size in pixels. -> 원하는 size대입

    4. **함수안에 함수를 정의하는 행동은 왜 하는 것일까?**   
        그것은 아래와 같은 상황에 사용한다. 함수안의 함수(fun2)는 fun1의 변수를 전역변수처럼 이용할 수 있다. 다시 말해 fun2는 a와 b를 매개변수로 받지 않았지만, 함수 안에서 a와 b를 사용하는 것을 확인할 수 있다. 
        ```python
        def fun1(self, a,b):
            a = 1
            b = 2
            def fun2(c):
                return a + b + c
            k = fun2(3)
            return k
        ```  
        
    5. 
        
            