---
layout: post
title: 【CV】Computer Vision at FastCampus1, chap1~6 
description: >
    FastCampus 사이트의 Computer vision 강의 내용 정리
---

1. FastCampus 사이트의 Computer vision 강의 내용을 정리해 두었다.
2. 수학적이고, 심도적인 내용은 윤성민교수님의 컴퓨터 비전 수업을 통해 배울 수 있었다. 강의 필기 자료는 다음에 업로드할 예정.
3. 심도적인 증명보다는, OpenCV를 사용해서 실습에 바로 적용할 수 있도록 하는데 중점을 주는 좋은 강의 였다. 
4. 몇가지 중요한 부분만 정리해 놓는다. 만약 다음에 공부할 필요가 느껴지면 그때 다시 사이트에 들어가서 공부하면 된다.
5. [FastCampus - Computer vision Lecture](https://fastcampus.co.kr/dev_online_cvodl)

# 목차

- 아래의 목차에서 정말 듣고 싶은 것만 듣자. 
- 어차피 학교 수업에서도 배울 내용이고, 너무 오래전 기술이라 몰라도 되는 것도 많다.
- 따라서 목차를 보고 '이런 내용이 있구나'하고 나중에 필요하면, 와서 듣자. 지금은 진짜 궁금한것만 듣자.
- ```sh
    OpenCV-Python 시작하기
    Ch 01. OpenCV-Python 시작하기 - 01. 전체 코스와 컴퓨터 비전 소개
    Ch 01. OpenCV-Python 시작하기 - 02. 영상의 구조와 표현
    Ch 01. OpenCV-Python 시작하기 - 03. OpenCV 소개와 설치
    Ch 01. OpenCV-Python 시작하기 - 04. VS Code 설치와 개발 환경 설정
    Ch 01. OpenCV-Python 시작하기 - 05. 영상 파일 불러와서 출력하기
    Ch 01. OpenCV-Python 시작하기 - 06. OpenCV 주요 함수 설명
    Ch 01. OpenCV-Python 시작하기 - 07. Matplotlib 사용하여 영상 출력하기
    Ch 01. OpenCV-Python 시작하기 - 08. 실전 코딩 - 이미지 슬라이드쇼
    OpenCV-Python 기초 사용법
    Ch 02. OpenCV-Python 기초 사용법 - 01. 영상의 속성과 픽셀 값 처리
    Ch 02. OpenCV-Python 기초 사용법 - 02. 영상의 생성, 복사, 부분 영상 추출
    Ch 02. OpenCV-Python 기초 사용법 - 03. 마스크 연산과 ROI
    Ch 02. OpenCV-Python 기초 사용법 - 04. OpenCV 그리기 함수
    Ch 02. OpenCV-Python 기초 사용법 - 05. 카메라와 동영상 처리하기 1
    Ch 02. OpenCV-Python 기초 사용법 - 06. 카메라와 동영상 처리하기 2
    Ch 02. OpenCV-Python 기초 사용법 - 07. 키보드 이벤트 처리하기
    Ch 02. OpenCV-Python 기초 사용법 - 08. 마우스 이벤트 처리하기
    Ch 02. OpenCV-Python 기초 사용법 - 09. 트랙바 사용하기
    Ch 02. OpenCV-Python 기초 사용법 - 10. 연산 시간 측정 방법
    Ch 02. OpenCV-Python 기초 사용법 - 11. 실전 코딩 - 동영상 전환 이펙트
    기본적인 영상 처리 기법
    Ch 03. 기본적인 영상 처리 기법 - 01. 영상의 밝기 조절
    Ch 03. 기본적인 영상 처리 기법 - 02. 영상의 산술 및 논리 연산
    Ch 03. 기본적인 영상 처리 기법 - 03. 컬러 영상 처리와 색 공간
    Ch 03. 기본적인 영상 처리 기법 - 04. 히스토그램 분석
    Ch 03. 기본적인 영상 처리 기법 - 05. 영상의 명암비 조절
    Ch 03. 기본적인 영상 처리 기법 - 06. 히스토그램 평활화
    Ch 03. 기본적인 영상 처리 기법 - 07. 특정 색상 영역 추출하기
    Ch 03. 기본적인 영상 처리 기법 - 08. 히스토그램 역투영
    Ch 03. 기본적인 영상 처리 기법 - 09. 실전 코딩 - 크로마키 합성
    필터링
    Ch 04. 필터링 - 01. 필터링 이해하기
    Ch 04. 필터링 - 02. 블러링 (1) - 평균값 필터
    Ch 04. 필터링 - 03. 블러링 (2) - 가우시안 필터
    Ch 04. 필터링 - 04. 샤프닝 - 언샤프 마스크 필터
    Ch 04. 필터링 - 05. 잡음 제거 (1) - 미디언 필터
    Ch 04. 필터링 - 06. 잡음 제거 (2) - 양방향 필터
    Ch 04. 필터링 - 07. 실전 코딩 - 카툰 필터 카메라
    기하학적 변환
    Ch 05. 기하학적 변환 - 01. 영상의 이동 변환과 전단 변환
    Ch 05. 기하학적 변환 - 02. 영상의 확대와 축소
    Ch 05. 기하학적 변환 - 03. 이미지 피라미드
    Ch 05. 기하학적 변환 - 04. 영상의 회전
    Ch 05. 기하학적 변환 - 05. 어파인 변환과 투시 변환
    Ch 05. 기하학적 변환 - 06. 리매핑
    Ch 05. 기하학적 변환 - 07. 실전 코딩 - 문서 스캐너
    영상의 특징 추출
    CH 06. 영상의 특징 추출 - 01. 영상의 미분과 소베 필터
    CH 06. 영상의 특징 추출 - 02. 그래디언트와 에지 검출
    CH 06. 영상의 특징 추출 - 03. 캐니 에지 검출
    CH 06. 영상의 특징 추출 - 04. 허프 변환 직선 검출
    CH 06. 영상의 특징 추출 - 05. 허프 원 변환 원 검출
    CH 06. 영상의 특징 추출 - 06. 실전 코딩 동전 카운터
    이진 영상 처리
    CH 07. 이진 영상 처리 - 01. 영상의 이진화
    CH 07. 이진 영상 처리 - 02. 자동 이진화 Otsu 방법
    CH 07. 이진 영상 처리 - 03. 지역 이진화
    CH 07. 이진 영상 처리 - 04. 모폴로지 (1) 침식과 팽창
    CH 07. 이진 영상 처리 - 05. 모폴로지 (2) 열기와 닫기
    CH 07. 이진 영상 처리 - 06. 레이블링
    CH 07. 이진 영상 처리 - 07. 외곽선 검출
    CH 07. 이진 영상 처리 - 08. 다양한 외곽선 함수
    CH 07. 이진 영상 처리 - 09. 실전 코딩 명함 인식 프로그램
    영상 분할과 객체 검출
    CH 08. 영상 분할과 객체 검출 - 01. 그랩컷
    CH 08. 영상 분할과 객체 검출 - 02. 모멘트 기반 객체 검출
    CH 08. 영상 분할과 객체 검출 - 03. 템플릿 매칭 (1) 이해하기
    CH 08. 영상 분할과 객체 검출 - 04. 템플릿 매칭 (2) 인쇄체 숫자 인식
    CH 08. 영상 분할과 객체 검출 - 05. 캐스케이드 분류기 - 얼굴 검출
    CH 08. 영상 분할과 객체 검출 - 06. HOG 보행자 검출
    CH 08. 영상 분할과 객체 검출 - 07. 실전 코딩 간단 스노우 앱
    특징점 검출과 매칭
    CH 09. 특징점 검출과 매칭 - 01. 코너 검출
    CH 09. 특징점 검출과 매칭 - 02. 특징점 검출
    CH 09. 특징점 검출과 매칭 - 03. 특징점 기술
    CH 09. 특징점 검출과 매칭 - 04. 특징점 매칭
    CH 09. 특징점 검출과 매칭 - 05. 좋은 매칭 선별
    CH 09. 특징점 검출과 매칭 - 06. 호모그래피와 영상 매칭
    CH 09. 특징점 검출과 매칭 - 07. 이미지 스티칭
    CH 09. 특징점 검출과 매칭 - 08. 실전 코딩 - AR 비디오 플레이어
    객체 추적과 모션 벡터
    CH 10. 객체 추적과 모션 벡터 - 01. 배경 차분 정적 배경 차분
    CH 10. 객체 추적과 모션 벡터 - 02. 배경 차분 이동 평균 배경
    CH 10. 객체 추적과 모션 벡터 - 03. 배경 차분 MOG 배경 모델
    CH 10. 객체 추적과 모션 벡터 - 04. 평균 이동 알고리즘
    CH 10. 객체 추적과 모션 벡터 - 05. 캠시프트 알고리즘
    CH 10. 객체 추적과 모션 벡터 - 06. 루카스 - 카나데 옵티컬플로우
    CH 10. 객체 추적과 모션 벡터 - 07. 밀집 옵티컬플로우
    CH 10. 객체 추적과 모션 벡터 - 08. OpenCV 트래커
    CH 10. 객체 추적과 모션 벡터 - 09. 실전 코딩 - 핸드 모션 리모컨
    머신러닝
    CH 11. 머신 러닝 - 01. 머신 러닝 이해하기
    CH 11. 머신 러닝 - 02. OpenCV 머신 러닝 클래스
    CH 11. 머신 러닝 - 03. k최근접 이웃 알고리즘
    CH 11. 머신 러닝 - 04. KNN 필기체 숫자 인식
    CH 11. 머신러닝 - 05. SVM 알고리즘
    CH 11. 머신러닝 - 06. OpenCV SVM 사용하기
    CH 11. 머신러닝 - 07. HOG SVM 필기체 숫자 인식
    CH 11. 머신러닝 - 08. 숫자 영상 정규화
    CH 11. 머신러닝 - 09. k-평균 알고리즘
    CH 11. 머신러닝 - 10. 실전 코딩 문서 필기체 숫자 인식
    딥러닝 이해와 영상 인식
    CH 12. 딥러닝 이해와 영상 인식 - 01. 딥러닝 이해하기
    CH 12. 딥러닝 이해와 영상 인식 - 02. CNN 이해하기
    CH 12. 딥러닝 이해와 영상 인식 - 03. 딥러닝 학습과 모델 파일 저장
    CH 12. 딥러닝 이해와 영상 인식 - 04. OpenCV DNN 모듈
    CH 12. 딥러닝 이해와 영상 인식 - 05. MNIST 학습 모델 사용하기
    CH 12. 딥러닝 이해와 영상 인식 - 06. GoogLeNet 영상 인식
    CH 12. 딥러닝 이해와 영상 인식 - 07. 실전 코딩 한글 손글씨 인식
    딥러닝 활용 : 객체 검출, 포즈 인식
    CH 13. 딥러닝 활용 객체 검출 - 01. OpenCV DNN 얼굴 검출
    CH 13. 딥러닝 활용 객체 검출 - 02. YOLOv3 객체 검출
    CH 13. 딥러닝 활용 객체 검출 - 03. Mask-RCNN 영역 분할
    CH 13. 딥러닝 활용 객체 검출 - 04. OpenPose 포즈 인식
    CH 13. 딥러닝 활용 객체 검출 - 05. EAST 문자 영역 검출
    CH 13. 딥러닝 활용 객체 검출 - 06. 실전 코딩 얼굴 인식
  ```



# Chap 1 - OpenCV basics

1강

- 책과 코드 : [https://github.com/sunkyoo/opencv4cvml](https://github.com/sunkyoo/opencv4cvml)
- OpenCV tutorial [https://docs.opencv.org/master/](https://docs.opencv.org/master/)
- 컴퓨터 비전 응용 분야 - 머신 비전(공장 자동화), 인식, 객체 검출, 화질 개선, 인공지능 서비스(테슬라, 아마존)



2강

- Gray (Black) 0~255 (White)
- 프로그래밍 언어의 1Byte
  - unsigned char
  - numpy.uint8
- 색의 성분 (**빛**의 삼원색)
  - 0 : 색의 성분이 전혀 없는 상태 = 검정
  - 255 : 색의 성분이 가득 있는 상태  = 흰색
- 영상의 좌표계
  - 행렬을 읽듯이 읽어야 한다. 
  - 우리가 아는 2차원 좌표계를 사용해서 읽지 말고. 따라서 y축은 아래를 향한다.
- 영상의 파일 형식의 특징
  - JPG는 원본 이미지 데이터가 약간 손실됨. 인간은 못 느끼는 정도
  - GIF는 움짤용. 영상 처리에서 사용 안함
  - PNG 영상처리에서 가장 많이 사용
  - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210111224950317.png?raw=true" alt="image-20210111224950317" style="zoom: 50%;" />



3강

- BSD license : Free for academic & commercial  use
- OpenCV 구성
  - [OpenCV main modules](https://github.com/opencv/opencv) 
  - [OpenCV extra modules](https://github.com/opencv/opencv_contrib/) - 새롭게 만들어지는 기능. 잘 사용되지 않는 것들. 라이센스 문제가 있어서 공유하면 안되는 것들.



4강 기본 코드

```python
import sys
import cv2
print(cv2.__version__)

img = cv2.imread('cat.bmp')

if img is None:
    print('Image load failed!')
    sys.exit()

cv2.namedWindow('image')
cv2.imshow('image', img)
cv2.waitKey() # 아무키나 누르면 window 종료
cv2.destroyAllWindows()
```



5강 OpenCV 기본 명령어

- OpenCV 도움말: [http://docs.opencv.org/](http://docs.opencv.org/) -> 버전 선택
- cv2.imread(filename, flags = 'cv2.IMREAD_COLOR ')  -> BGR 순서
- 이미지 저장 : cv2.imwrite(filename, img : numpy.ndarray, params=None) 
- 새 창 띄우기(imshow위해 꼭 안해도 됨) : cv2.namedWindow(winname :  str, flags=None) -> None
- 새 창 모두 닫기 : cv2.destroyWindow(winname : str) 
- cv2.moveWindow(winname, x, y) , cv2.resizeWindow(winname, width, height) 
- cv2.imshow(winname, mat : ndarray_dtype_8uint) 
- cv2.waitKey(delay='0')
  - dalay = ms 단위의 대기 시간.  If delay = 0, 무한히 기다림
  - return 값은 눌린 키 값.
  - 사용법    
    ```python
    while True:
    	if cv2.waitKey() == ord('q'):
    		break
    # 27(ESC), 13(ENTER), 9(TAB)
    ```



6강 Matplotlib 기본 명령어

- BGR 순서 -> cv2.cvtColor() , GRAY -> cmap='gray'   

  ```python
  import matplotlib.pyplot as plt
  import cv2
  
  # 컬러 영상 출력
  imgBGR = cv2.imread('cat.bmp')
  imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB) # cv2.IMREAD_GRAYSCALE
  
  plt.axis('off')
  plt.imshow(imgRGB) #  cmap='gray'
  plt.show()
  
  # subplot
  plt.subplot(121), plt.axis('off'), plt.imshow(imgRGB)
  plt.subplot(122), plt.axis('off'), plt.imshow(imgGray, cmap='gray')
  plt.show()
  ```



8강 - 이미지 슬라이드쇼 프로그램
- 폴더에서 파일 list 가져오기  
  ```python
  import os
  file_list = os.listdir('.\\images')
  img_files = [file for file in file_list if file.endswith('.jpg')]
  # 혹은
  import glob
  img_files = glob.glob('.\\images\\*.jpg') 
  ```
- 전체 화면 영상 출력

  ```python
  cv2.namedWindow('image', cv2.WINDOW_NORMAL) # normal 이여야 window 창 변경 가능.
  cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
  ```
- 반복적으로, 이미지 출력  
  ```python
  cnt = len(img_files)
  idx = 0
  while True:
      img = cv2.imread(img_files[idx])
      if img is None:
          print('Image load failed!')
          break
      cv2.imshow('image', img)
      if cv2.waitKey(1000) >= 27: # 1초 지나면 False, ESC누루면 True
          break # while 종료
      # cv2.waitKey(1000) -> 1
      # cv2.waitKey(1000) >= 0: 아무키나 누르면 0 초과의 수 return
      idx += 1
      if idx >= cnt:
      	idx = 0
  ```

  

# chap2 - Basic image processing technique

1장
- cv2.imread('cat.bmp', cv2.IMREAD_COLOR) -> numpy.ndarray **[행, 열, depth, 몇장]**
- ```python
  for y in range(h):
      for x in range(w):
          img1[y, x] = 255
          img2[y, x] = [0, 0, 255]
  ```
- 이런 식으로 이미지, 픽셀값을 **For로 바꾸면 엄청느리다**. 
- 함수를 찾아 보거나, img[:,:,:] = [0,255,255] 이런식으로 사용해야 한다. 

2장
- 이미지(=영상) 새로 생성하기
  
  - numpy.empty / numpy.zeros / numpy.ones / numpy.full
- 이미지 복사  
    - ```python
        img1 = cv2.imread('HappyFish.jpg')
        img2 = img1 -> 같은 메모리 공유
        img3 = img1.copy() 
      ```
    - img3 = img1.copy() -> 새로운 메모리에 이미지 정보 다시 할당 array안의 array도 다시 할당한다. 여기서는 deepcopy랑 같다. [](추가 설명)](https://junha1125.github.io/docker-git-pytorch/2021-01-07-torch_module_research/#21-copydeepcopy)
    - **numpy에서는 deepcopy랑 copy가 같다.** 라고 외우자



### 2-3강 - 마스크 연산과 ROI
- 마스크 영상으로는 0 또는 255로 구성된 이진 영상(binary image), Gray Scale
- cv2.copyTo(src, mask, dst=None) -> dst
    - ```python
        src = cv2.imread('airplane.bmp', cv2.IMREAD_COLOR)
        mask = cv2.imread('mask_plane.bmp', cv2.IMREAD_GRAYSCALE)
        dst = cv2.imread('field.bmp', cv2.IMREAD_COLOR)
        cv2.copyTo(src, mask, dst1)
        dst2 = cv2.copyTo(src, mask)
        
        # 하지만 아래와 같은 슬라이딩 연산도 가능!!
        dst[mask > 0] = src[mask > 0] # -> dist = dst1
      ```
    - src, mask, dst는 w,h 모두 크기가 같아야 함. src와 dst는 같은 타입. mask는 그레이스케일 타입의 이진 영상.
    - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210113090901294.png?raw=true" alt="image-20210113090901294" style="zoom:80%;" />
    - dst2 : <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210113090939395.png?raw=true" alt="image-20210113090939395" style="zoom: 33%;" />
    - dst과 dst2는 완전히 다른 이미지로 생성되니 주의해서 사용할 것. 
- 투명한 배경이 있는 png 파일 (4channel)
  - ```python
    src = cv2.imread('airplane.bmp', cv2.IMREAD_UNCHANGED) ## 투명배경 있는 것은 IMREAD_UNCHANGED!!! 
    mask = src[:,:,-1]
    src = src[;,:,0:3]
    h,w = src.shape[:2]
    crop = dst[x:x+h,y:w+2]  # src, mask, dst는 w,h 모두 크기가 같아야 함
    cv2.copyTo(src, mask, crop)
    ```

### 2-4강 - OpenCV그리기 함수
- 주의할 점 : in-place 알고리즘 -> 원본 데이터 회손
- 자세한 내용은 인강 참조, 혹은 OpenCV 공식 문서 참조 할 것.
- 직선 그리기 : cv2.line
- 사각형 그리기 : cv2.rectangle
- 원 그리기 : cv2.circle
- 다각형 그리기 : cv2.polylines
- 문자열 출력 : cv2.putText 

### 5강 - 카메라와 동영상 처리하기 1
- OpenCV에서는 카메라와 동영상으로부터 프레임(frame)을 받아오는 작업을 **cv2.VideoCapture** 클래스 하나로 처리함  
  <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210113093215343.png?raw=true" alt="image-20210113093215343" style="zoom: 80%;" />
- 카메라 열기
  - index : 2대가 있으면 테스트 해봐서 0,1번 중 하나 이다. domain_offset_id는 무시하기(카메라를 Open하는 방법을 운영체제가 적절한 방법을 사용한다.) 
  - 아래 코드 꼭 읽어 보기:
    <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210113093306495.png?raw=true" alt="image-20210113093306495" style="zoom:80%;" />
  - 추가 함수
    - 준비 완료? : cv2.VideoCapture.isOpened() -> retval
    - 프래임 받아오기 : cv2.VideoCapture.read(image=None) -> retval, image
  - ```python
      import sys
      import cv
      cap = cv2.VideoCapture()
      cap.open(0)
      # 혹은 위의 2줄 한줄로 cap = cv2.VideoCapture(0)
      
      if not cap.isOpend()
        print('camera open failed!')
          sys.exit()
      while Ture :
          ret, frame = cap.read()
          if not ret : # frame을 잘 받아 왔는가? # 가장 마지막 프레임에서 멈춘다
              break
          edge = cv2.Canny(frame,50,150)
          cv2.imshow('edge', edge) # 2개의 창이 뜬다!! figure 설정할 필요 없음
          cv2.imshow('frame', frame)
          cv2.waitKey(20) == 27: # ESC눌렀을떄
              break
      
      cap.release() # cap 객체 delete
      cv2.destroyAllWindows()
    ```
- 동영상 열기
  
  - 위의 코드에서 cap = cv2.VideoCapture('Video File Path') 를 넣어주고 나머지는 위에랑 똑같이 사용하면 된다. 
- 카메라 속성 열고 바꾸기
  - cv2.VideoCapture.get(propId) -> cap.get('propId')
  - cv2.VideoCapture.set(propId, value) -> cap.set('propId', value)



### 2-6장 - 카메라와 동영상 처리하기 2
- 동영상 저장 : cv2.VideoWriter 클래스
  - 소리는 저장이 안된다!! 
  - cv2.VideoWriter / cv2.VideoWriter.open
  - cv2.VideoWriter.isOpened() / cv2.VideoWriter.write(image)
  - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210113094959326.png?raw=true" alt="image-20210113094959326" style="zoom: 80%;" />
- 코덱 종류와 다운로드 링크
  
  - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210113094922834.png?raw=true" alt="image-20210113094922834" style="zoom:80%;" />
- 예시 코드
  - 이미지 반전 :  inversed = ~frame. RGB 중 Grean = 0, RB는 255에 가까워짐
  - ```python
    # 나의 카메로 open
    cap = cv2.VideoCapture(0)
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 동영상 준비 중
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') # *'DIVX' == 'D','I','V','X'
    fps = 30
    delay = round(1000/fps) # 프레임과 프레임 사이의 시간간격 (1000ms/30fps)
    out = cv2.VideoWriter('output.avi', fourcc, 30, (w, h))
    
    if not out.isOpened():
        print('File open failed!')
        cap.release()
        sys.exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        inversed = ~frame 
        # 반전하는 코드. RGB 중 Grean = 0, RB는 255에 가까워짐
        # 신기하니, 아래의  cv2.imshow('inversed', inversed) 결과 확인해보기.
        
        """
        edge = cv2.Canny(frame,50,150)
        edge_color = cv2.cvtColor(edge,cv2.COLOR_GRAY2BGR)
        out.write(edge_color)
        """
        
        out.write(frame)
        cv2.imshow('frame', frame) 
        cv2.imshow('inversed', inversed)
        if cv2.waitKey(delay) == 27: # delay는 이와 같이 사용해도 좋다. 없어도 됨.
        	break	
    
    ```

### 2-7장 - 키보드 이벤트 처리하기
- cv2.waitKey(delay=None) -> retval
- while True: 문을 계속 돌면서, 매 시간 마다 키보드 input이 없으면 필요없는 값을 return하고 while문에는 영향을 끼치지 않는다.
- ```python
  # 키보드에서 'i' 또는 'I' 키를 누르면 영상을 반전
  import cv2
  img = cv2.imread('cat.bmp', cv2.IMREAD_GRAYSCALE)
  cv2.imshow('image', img)
  while True:
      keycode = cv2.waitKey()
      if keycode == ord('i') or keycode == ord('I'):
          img = ~img
          cv2.imshow('image', img)
      elif keycode == 27:
          break
  cv2.destroyAllWindows()
  ```

### 2-8장 - 마우스 이벤트 처리하기
- 마우스 이벤트 콜백함수 등록 함수 : cv2.setMouseCallback(windowName, onMouse, param=None) -> None
- 마우스 이벤트 처리 함수(콜백 함수) 형식 : onMouse(event, x, y, flags, param) -> None
- 이벤트에 대한 event 목록들은 강의 자료 참조.
- ```python
  oldx = oldy = -1
  def on_mouse(event, x, y, flags, param):
      global oldx, oldy
      if event == cv2.EVENT_LBUTTONDOWN:
      	oldx, oldy = x, y
      elif event == cv2.EVENT_MOUSEMOVE:
      	if flags & cv2.EVENT_FLAG_LBUTTON:
              cv2.line(img, (oldx, oldy), (x, y), (0, 0, 255), 4, cv2.LINE_AA)
              cv2.imshow('image', img)
              oldx, oldy = x, y
  img = np.ones((480, 640, 3), dtype=np.uint8) * 255
  cv2.imshow('image', img)
  cv2.setMouseCallback('image', on_mouse)
  cv2.waitKey()
  ```

### 2-9강 - 트랙바 사용하기

- cv2.createTrackbar(trackbarName, windowName, value, count, onChange) -> None
- ```python
  def on_level_change(pos):
      value = pos * 16
      if value >= 255:
      	value = 255
      img[:] = value
      cv2.imshow('image', img)
  img = np.zeros((480, 640), np.uint8)
  cv2.namedWindow('image')
  cv2.createTrackbar('level', 'image', 0, 16, on_level_change)
  cv2.imshow('image', img)
  cv2.waitKey()
  cv2.destroyAllWindows()
  
  ```



10장 - 연산시간 측정
- python time.time() 사용하자.
- cv2의 시간측정함수를 소개한다.

### 2-11장 - 동영상 전환 이펙트 코드 만들기
- <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210114084035017.png?raw=true" alt="image-20210114084035017" style="zoom:80%;" />
-   
    ```python
        import sys
        import numpy as np
        import cv2
    ```


~~~python
    # 두 개의 동영상을 열어서 cap1, cap2로 지정
    cap1 = cv2.VideoCapture('video1.mp4')
    cap2 = cv2.VideoCapture('video2.mp4')

    if not cap1.isOpened() or not cap2.isOpened():
        print('video open failed!')
        sys.exit()

    # 두 동영상의 크기, FPS는 같다고 가정함
    frame_cnt1 = round(cap1.get(cv2.CAP_PROP_FRAME_COUNT))  # 15초 * 24 = Total 360 frame
    frame_cnt2 = round(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap1.get(cv2.CAP_PROP_FPS) # 24
    effect_frames = int(fps * 2)  # 48 -> 1번 동영상의 맨 뒤 48프레임과, 2번 동영상의 맨 앞 48프레임이 겹친다

    print('frame_cnt1:', frame_cnt1)
    print('frame_cnt2:', frame_cnt2)
    print('FPS:', fps)

    delay = int(1000 / fps)

    w = round(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    # 출력 동영상 객체 생성
    out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))

    # 1번 동영상 복사
    for i in range(frame_cnt1 - effect_frames):
        ret1, frame1 = cap1.read()

        if not ret1:
            print('frame read error!')
            sys.exit()

        out.write(frame1)
        print('.', end='')

        cv2.imshow('output', frame1)
        cv2.waitKey(delay)

    # 1번 동영상 뒷부분과 2번 동영상 앞부분을 합성
    for i in range(effect_frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print('frame read error!')
            sys.exit()

        dx = int(w / effect_frames) * i

        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, 0:dx, :] = frame2[:, 0:dx, :]
        frame[:, dx:w, :] = frame1[:, dx:w, :]

        #alpha = i / effect_frames
        #frame = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)

        out.write(frame)
        print('.', end='')

        cv2.imshow('output', frame)
        cv2.waitKey(delay)

    # 2번 동영상을 복사
    for i in range(effect_frames, frame_cnt2):
        ret2, frame2 = cap2.read()

        if not ret2:
            print('frame read error!')
            sys.exit()

        out.write(frame2)
        print('.', end='')

        cv2.imshow('output', frame2)
        cv2.waitKey(delay)

    print('\noutput.avi file is successfully generated!')

    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()

```
~~~



# chap3 - 기본 영상 처리 기법

Ch 03. 기본적인 영상 처리 기법 - 01. 영상의 밝기 조절

- 밝기 조절을 위한 함수 : **cv2.add**(src1, src2, dst=None, mask=None, dtype=None) -> dst



Ch 03. 기본적인 영상 처리 기법 - 02. 영상의 산술 및 논리 연산

- 두 이미지 덧셉, 가중치 합
- **cv2.add**(src1, src2, dst=None, mask=None, dtype=None) -> dst
- cv2.addWeighted(src1, alpha, src2, beta, gamma, dst=None, dtype=None) -> dst
- 두 이미지 뺄셈
- **cv2.subtract**(src1, src2, dst=None, mask=None, dtype=None) -> dst
- 두 이미지 차이 계산  dst( , ) = \|src1( , ) -  src2( , ) \|
- **cv2.absdiff**(src1, src2, dst=None) -> dst



Ch 03. 기본적인 영상 처리 기법 - 03. 컬러 영상 처리와 색 공간

- b_plane, g_plane, r_plane = cv2.split(src)
- b_plane = src[:, :, 0] g_plane = src[:, :, 1] r_plane = src[:, :, 2]
- **cv2.cvtColor**(src, code, dst=None, dstCn=None) -> dst



Ch 03. 기본적인 영상 처리 기법 - 04. 히스토그램 분석

- **cv2.calcHist(**images, channels, mask, histSize, ranges, hist=None, accumulate=None) -> hist

- ```python
  src = cv2.imread('lenna.bmp')
  colors = ['b', 'g', 'r']
  bgr_planes = cv2.split(src)
  for (p, c) in zip(bgr_planes, colors):
      hist = cv2.calcHist([p], [0], None, [256], [0, 256])
      plt.plot(hist, color=c)
  plt.show()
  ```



Ch 03. 기본적인 영상 처리 기법 - 05. 영상의 명암비 조절

- dst( , ) = saturate(src( , ) +  (src( , ) - 128) ) + a
- <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210114085414553.png?raw=true" alt="image-20210114085414553" style="zoom:50%;" />
- ```python
    src = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)
    alpha = 1.0
    dst = np.clip((1+alpha)*src - 128*alpha, 0, 255).astype(np.uint8)
  ```
- 명암비 자동 조절
  
  - dst = **cv2.normalize**(src, None, 0, 255, cv2.NORM_MINMAX)



Ch 03. 기본적인 영상 처리 기법 - 06. 히스토그램 평활화

- 히스토그램이 그레이스케일 전체 구간에서 균일한 분포로 나타나도록 변경하는 명암비 향상 기법 
- 히스토그램 균등화, 균일화, 평탄화
- **cv2.equalizeHist**(src, dst=None) -> dst



Ch 03. 기본적인 영상 처리 기법 - 07. 특정 색상 영역 추출하기

- 특정 범위 안에 있는 행렬 원소 검출 : **cv2.inRange**(src, lowerb, upperb, dst=None) -> dst



Ch 03. 기본적인 영상 처리 기법 - 08. 히스토그램 역투영

- **cv2.calcBackProjec**t(images, channels, hist, ranges, scale, dst=None) -> dst
- <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210114090121180.png?raw=true" alt="image-20210114090121180" style="zoom:50%;" />



Ch 03. 기본적인 영상 처리 기법 - 09. 실전 코딩 - 크로마키 합성

- cv2.inRange() 함수를 사용하여 50 ≤ 𝐻 ≤ 80, 150 ≤ 𝑆 ≤ 255, 0 ≤ 𝑉 ≤ 255 범위의 영역을 검출
- 마스크 연산을 지원하는 cv2.copyTo() 함수 사용
- <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210114090209199.png?raw=true" alt="image-20210114090209199" style="zoom:50%;" />
- 강의자료 FastCampus_CV\opencv_python_ch01_ch05\ch03\chroma_key.py 참조



# chap4 -  필터링

- **cv2.filter2D**

- **cv2.GaussianBlur**

- **cv2.medianBlur**(src, ksize, dst=None) -> dst

  - 주변 픽셀들의 값들을 정렬하여 그 중앙값(median)으로 픽셀 값을 대체 
  - 소금-후추 잡음 제거에 효과적

- **cv2.bilateralFilter**(src, d, sigmaColor, sigmaSpace, dst=None, borderType=None) -> dst

  - edge-preserving noise removal filter / Bilateral filter
  - 평균 값 필터 또는 가우시안 필터는 에지 부근에서도 픽셀 값을 평탄하게 만드는 단점
  - 에지가 아닌 부분에서만 blurring(잡음제거)

- 카툰 필터 만들기

  - 아래의 방법은 사람들이 찾아낸 방법 중 하나. 다양한 방법이 있다.

  - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210114090945601.png?raw=true" alt="image-20210114090945601" style="zoom: 50%;" />

  - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210114091006607.png?raw=true" alt="image-20210114091006607" style="zoom:50%;" />

  - ```python
    # 카툰 필터 카메라
    
    import sys
    import numpy as np
    import cv2
    
    def cartoon_filter(img):
        h, w = img.shape[:2]
        img2 = cv2.resize(img, (w//2, h//2))
        blr = cv2.bilateralFilter(img2, -1, 20, 7)
        edge = 255 - cv2.Canny(img2, 80, 120)
        edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        dst = cv2.bitwise_and(blr, edge)  # 논리 연산자
        dst = cv2.resize(dst, (w, h), interpolation=cv2.INTER_NEAREST)
        return dst
    
    def pencil_sketch(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blr = cv2.GaussianBlur(gray, (0, 0), 3)
        dst = cv2.divide(gray, blr, scale=255)
        return dst
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('video open failed!')
        sys.exit()
    cam_mode = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cam_mode == 1:
            frame = cartoon_filter(frame)
        elif cam_mode == 2:
            frame = pencil_sketch(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord(' '):
            cam_mode += 1
            if cam_mode == 3:
                cam_mode = 0
    
    cap.release()
    cv2.destroyAllWindows()
    
    ```



# chap5 -  기하학적 변환

- 수학적 공식은 '20년2학기/윤성민 교수님 컴퓨터 비전 수업 자료 참조'

- **cv2.warpAffine**(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None) -> dst

- **cv2.resize**(src, dsize, dst=None, fx=None, fy=None, interpolation=None) -> dst

- **cv2.flip**(src, flipCode, dst=None) -> dst

- **cv2.pyrDown**(src, dst=None, dstsize=None, borderType=None) -> dst : 다중 피라미드 이미지 자동 return

- **cv2.pyrUp**(src, dst=None, dstsize=None, borderType=None) -> dst

- **rot = cv2.getRotationMatrix2D**(cp, 20, 1)

  - **dst = cv2.warpAffine**(src, rot, (0, 0))

- 어파인 변환과 투시 변환

  - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210114091721057.png?raw=true" alt="image-20210114091721057" style="zoom: 80%;" />

  - **cv2.getAffineTransform(src, dst)** -> retval

  - **cv2.getPerspectiveTransform**(src, dst, solveMethod=None) -> retval

  - **cv2.warpAffine**(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None) -> dst

  - **cv2.warpPerspective**(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None) -> dst

  - ```python
    src = cv2.imread('namecard.jpg')
    w, h = 720, 400
    srcQuad = np.array([[325, 307], [760, 369], [718, 611], [231, 515]],
    np.float32)
    dstQuad = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)
    pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
    dst = cv2.warpPerspective(src, pers, (w, h))
    ```

  - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210114091945757.png?raw=true" alt="image-20210114091945757" style="zoom: 67%;" />

- 리매핑 : 영상의 특정 위치 픽셀을 다른 위치에 재배치하는 일반적인 프로세스

  - **cv2.remap**(src, Pixel좌표1, Pixel좌표2, interpolation, dst=None, borderMode=None, borderValue=None) -> dst

- [실전 코딩] 문서 스캐너

  - 위의 방법 적절히 사용
  - 코드는 FastCampus_CV\opencv_python_ch01_ch05\ch03\docuscan.py 참조



# chap6 - 영상의 특징 추출

- 아래의 함수를 사용하고 싶다면, 어줍잖은 블로급 보지 말고 동영상 찾아보자.
- **cv2.Sobel**(src, ddepth, dx, dy, dst=None, ksize=None, scale=None, delta=None, borderType=None) -> dst
- cv2.Scharr(src, ddepth, dx, dy, dst=None, scale=None, delta=None, borderType=None) -> dst
- 2D 벡터의 크기 계산 함수 : **cv2.magnitude**(x, y, magnitude=None) -> magnitude
- 2D 벡터의 방향 계산 함수 : cv2.phase(x, y, angle=None, angleInDegrees=None) -> angle
- **cv2.Canny**(image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None) -> edges
- cv2.HoughLines(image, rho, theta, threshold, lines=None, srn=None, stn=None, min_theta=None, max_theta=None) -> lines
- **cv2.HoughLinesP**(image, rho, theta, threshold, lines=None, minLineLength=None, maxLineGap=None) -> lines
- **cv2.HoughCircles**(image, method, dp, minDist, circles=None, param1=None, param2=None, minRadius=None, maxRadius=None) -> circles

























