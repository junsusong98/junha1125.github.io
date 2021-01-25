---
layout: post
title: 【CV】Computer Vision at FastCampus 2
---

1.  FastCampus 사이트의 Computer vision 강의 내용 정리
2.  **<u>구글링을 해도 되지만은, 필요하면 강의를 찾아서 듣기</u>**
3.  [FastCampus - Computer vision Lecture](https://fastcampus.co.kr/dev_online_cvodl)
4.  [이전 Post Link](https://junha1125.github.io/self-study/2021-01-13-fast_campus1/)



# chap7 - Binary 

1. **cv2.threshold**(src, thresh, maxval, type, dst=None) -> retval(사용된 임계값), dst

2. Otsu 방법

   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210119095215096.png?raw=tru" alt="image-20210119095215096" style="zoom:67%;" />
   - th, dst = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

3. 균일하지 않은 조명 환경 - 픽셀 주변에 작은 윈도우를 설정하여 지역 이진화 수행

   - ```python
     for y in range(4):
         for x in range(4):
             src_ = src[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
             dst_ = dst2[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
             cv2.threshold(src_, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU, dst_)
     ```

   - **cv2.adaptiveThreshold**(src, maxValue, adaptiveMethod, thresholdType, blockSize(슬라이딩 윈도우의 크기), C, dst=None) -> dst

4. 모폴로지(Morphology, 침식과 팽창)

   1. 침식 연산 : 영역이 줄어듦. 잡음 제거 효과 - **cv2.erode(src, kernel)**

   2. 팽창 연산 : 영역이 불어남. 구멍이 채워짐 - **cv2.dilate(src, kernel)** 

   3.  kernerl 생성 방법 : **cv2.getStructuringElement**

   4. ```python
      src = cv2.imread('circuit.bmp', cv2.IMREAD_GRAYSCALE)
      
      se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
      
      dst1 = cv2.erode(src, se)
      dst2 = cv2.dilate(src, None)
      ```

5. 모폴로지(열기와 닫기)

   1. 열기 : 침식 -> 팽창
   2. 닫기 : 팽창 -> 침식
   3. ![image-20210119100131147](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210119100131147.png?raw=tru)
   4. 범용 모폴로지 연산 함수 : **cv2.morphologyEx**(src, op, kernel)
   5. 열기 연산을 이용한 잡음 제거 : **우선은 지역 이진화!! 필수 -> 그리고 열기 연산**

6. 레이블링

   - 객체 분활 클러스터링(Connected Component Labeling / Contour Tracing)
   - 4-neightbor connectivity / 8-neightbor connectivity
   - 레이블링 함수 : **cv2.connectedComponents**(image)
   - 객체 정보 함께 반환하는 레이블링 함수 : **cv2.connectedComponentsWithStats**(image)
     - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210119100708533.png?raw=tru" alt="image-20210119100708533" style="zoom:67%;" />
     - 바운딩 박스 정보가 나오므로, 숫자 검출 같은 행위가 가능해 진다. 

7. 외곽선 검출( Boundary tracking. Contour tracing)

   - **cv2.findContours(image, mode, method)**

   - **cv2.drawContours(image, contours, contourIdx, color)** : 외각 선만 그려줌 (내부x)

   - ```python
     src = cv2.imread('contours.bmp', cv2.IMREAD_GRAYSCALE)
     #contours, hier = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
     contours, hier = cv2.findContours(src, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
     dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
     idx = 0
     while idx >= 0:
         c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
         cv2.drawContours(dst, contours, idx, c, 2, cv2.LINE_8, hier)
         idx = hier[0, idx, 0]
     ```

   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210119103238413.png?raw=tru" alt="image-20210119103238413" style="zoom: 67%;" />

   - **외각선 검출 및 도형의 크기나 특징 정보 반환**하는 함수

     - | 함수 이름                  | 설명                                                  |
       | :------------------------- | :---------------------------------------------------- |
       | cv2.arcLength()            | 외곽선 길이를 반환                                    |
       | cv2.contourArea()          | 외곽선이 감싸는 영역의 면적을 반환                    |
       | cv2.boundingRect()         | 주어진 점을 감싸는 최소 크기 사각형(바운딩 박스) 반환 |
       | cv2.minEnclosingCircle()   | 주어진 점을 감싸는 최소 크기 원을 반환                |
       | cv2.minAreaRect()          | 주어진 점을 감싸는 최소 크기 회전된 사각형을 반환     |
       | cv2.minEnclosingTriangle() | 주어진 점을 감싸는 최소 크기 삼각형을 반환            |
       | **cv2.approxPolyDP()**     | **외곽선을 근사화(단순화) - 아래 실습에서 사용 예정** |
       | cv2.fitEllipse()           | 주어진 점에 적합한 타원을 반환                        |
       | cv2.fitLine()              | 주어진 점에 적합한 직선을 반환                        |
       | cv2.isContourConvex()      | 컨벡스인지를 검사                                     |
       | cv2.convexHull()           | 주어진 점으로부터 컨벡스 헐을 반환                    |
       | cv2.convexityDefects()     | 주어진 점과 컨벡스 헐로부터 컨벡스 디펙트를 반환      |

8. 다각형 검출 프로그램 실습하기

   - 구현 순서
     1. 이진화 
     2. contour 
     3. 외각선 근사화  
     4. 너무 작은 객체, 컨벡스가 아닌 개체 제외 
     5. 꼭지점 개수 확인 (사각형, 삼각형, 원 검출)
   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210119104254897.png?raw=tru" alt="image-20210119104254897" style="zoom:80%;" />

9. 실전 코딩 : 명함 인식 프로그램 만들기

   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210119104456932.png?raw=tru" alt="image-20210119104456932"  />

   - 코드 핵심 요약   

     ```python
     import sys
     import numpy as np
     import cv2
     import pytesseract
     
     # 입력 영상 전처리
     src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
     _, src_bin = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
     # 외곽선 검출 및 명함 검출
     contours, _ = cv2.findContours(src_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
     
     for pts in contours:
         # 외곽선 근사화
         approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True)*0.02, True)
         # 컨벡스가 아니고, 사각형이 아니면 무시
         if not cv2.isContourConvex(approx) or len(approx) != 4:
             continue
         cv2.polylines(cpy, [approx], True, (0, 255, 0), 2, cv2.LINE_AA)
         
     pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
     dst = cv2.warpPerspective(src, pers, (dw, dh))
     dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
     print( pytesseract.image_to_string(dst_gray, lang='Hangul+eng') )
     ```

   - [Tesseract(광학 문자 인식(OCR) 라이브러리)](https://github.com/tesseract-ocr/tesseract)

     - 2006년부터 구글(Google)에서 관리. 현재는 2018년 이후 LSTM 기반 OCR 엔진 및 모델 추가. 
     - 하지만 우리는 github 그대로 사용하지 않을 것. 이미 빌드된 실행 파일 사용할 것
       - [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
       - tesseract/tesseract-ocr-w64-setup-v5.0.0-alpha.20200328.exe 다운 및 **설치**
     - 설치 옵션 및 설정 (이해 안되면 동영상 참고 하기)
       - **설치** 시 "Additional script data" 항목에서 "Hangul Script", "Hangul vertical script" 항목 체크, "Additional language data" 항목에서 "Korean" 항목 체크 
       - 설치 후 시스템 환경변수 PATH에 Tesseract 설치 폴더 추가 (e.g.) **c:\Program Files\Tesseract-OCR**
       -  <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210119110127435.png?raw=tru" alt="image-20210119110127435" style="zoom: 50%;" />
       - (안해도 됨) 설치 후 시스템 환경변수에 TESSDATA_PREFIX를 추가하고, 변수 값을 \tessdata 로 설정 
       - \tessdata\script\ 폴더에 있는 **Hangul.traineddata, Hangul_vert.traineddata 파일**을 \tessdata\ 폴더로 복사
       - 그리고$ pip install pytesseract
     - python : **pytesseract.image_to_string(dst_gray(np.array, gray, RGB도 가능, BGR 불가), lang='Hangul+eng')**
     - 그리고 가능하면, CMD에서 (path가 지정되어 있으므로) python namecard.py 로 실행하기

     

# chap8 - Segmentation & Detection

1. 그랩컷 영상분할

   - 그래프 알고리즘을 이용해서 영상 분할을 수행하는 알고리즘 (정확한 알고리즘은 [논문 참조](https://grabcut.weebly.com/background--algorithm.html))
   - **cv2.grabCut(img, mask, rect)**  
     **mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')**  
     **dst = src * mask2[:, :, np.newaxis]**
   - 마우스를 활용한 그랩컷 영상 분할 예제 : grabcut2.py (강의에서도 보라고 조언만 함) 

2. 모멘트 기반 (비슷한 모양 찾기 기법을 이용한) 객체 검출

   - <img src="C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210125090819511.png" alt="image-20210125090819511" style="zoom:80%;" />
   - Hu's seven invariant moments : 크기, 회전, 이동, 대칭 변환에 불변
   - 모양 비교 함수: **cv2.matchShapes(contour1, contour2, method, parameter)** -> 영상 사이의 거리(distance)

3. 템플릿 매칭

   - 입력영상에서 작은 크기의 템플릿과 일치하는 부분 찾는 기법

   - <img src="C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210125091301957.png" alt="image-20210125091301957" style="zoom:80%;" />

   - **cv2.matchTemplate(image, templ, method, result=None, mask=None) -> result**   
     image의 크기가 W x H 이고, templ의 크기가 w x h 이면 result 크기는 (W - w + 1) x (H - h +1)

   - method 부분에 들어가야할, distance 구하는 수식은 강의 및 강의 자료 참조

   - ```python
     res = cv2.matchTemplate(src, templ, cv2.TM_CCOEFF_NORMED)
     res_norm = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
     _, maxv, _, maxloc = cv2.minMaxLoc(res)
     ```

4. 템플릿 매칭 (2) - 인쇄체 숫자 인식

   - <img src="C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210125092302957.png" alt="image-20210125092302957" style="zoom:80%;" />
   - 오른쪽의 0~9까지는 미리 파일로 저장해놓음
   - 자세한 코드 사항은 강의 및 digitrec.py파일 참조

5. 캐스케이드 분류기: 얼굴 검출

   - Viola - Jones 얼굴 검출기 (이것도 머신러닝 기반)

     - 유사 하르 특징(Haar-like features)

   - [**Cascade Classifier OpenCV document**](https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html), [얼굴 검출 시각화 Youtube](https://www.youtube.com/watch?v=hPCTwxF0qf4)

   - **cv2.CascadeClassifier.detectMultiScale(image)**

   - [미리 학습된 XML 파일 다운로드](https://github.com/opencv/opencv/tree/master/data/haarcascades)

   - ```python
     src = cv2.imread('lenna.bmp')
     classifier = cv2.CascadeClassifier()
     classifier.load('haarcascade_frontalface_alt2.xml')
     faces = classifier.detectMultiScale(src)
     for (x, y, w, h) in faces:
         face_img = src[y:y+h, x:x+w]
         cv2.rectangle(src, (x, y, w, y), (255, 0, 255), 2)
     ```

6. HOG 보행자 검출

   - Histogram of Oriented Gradients, 지역적 그래디언트 방향 정보를 특징 벡터로 사용. SIFT에서의 방법을 최적화하여 아주 잘 사용한 방법

   - 2005년부터 한동안 가장 좋은 방법으로, 다양한 객체 인식에서 활용되었다.

   - <img src="C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210125093630644.png" alt="image-20210125093630644" style="zoom: 80%;" />

     - 9개 : 180도를 20도 단위로 나눠서 9개 단위로 gradient 분류
     - 1개 셀 8x8, 1개 블록 16 x 16. 블록 1개 는 36개의 히스토그램 정보를 가짐

   - **cv2.HOGDescriptor.detectMultiScale(img)**   
     **hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())**

   - ```python
     cap = cv2.VideoCapture('vtest.avi')
     hog = cv2.HOGDescriptor()
     hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
     while True:
         ret, frame = cap.read()
         detected, _ = hog.detectMultiScale(frame)
         for (x, y, w, h) in detected:
         c = (random.randint(0, 255), random.randint(0, 255),
         random.randint(0, 255))
         cv2.rectangle(frame, (x, y), (x + w, y + h), c, 3)
     ```

7. 실전 코딩: 간단 스노우앱

   - 구현 기능

     - 카메라 입력 영상에서 얼굴&눈 검출하기 (캐스케이드 분류기 사용)
     - 눈 위치와 맞게 투명한 PNG 파일 합성하기 
     - 합성된 결과를 동영상으로 저장하기

   - ch8/snowapp.py 파일 참조  

     ```python
     face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
     eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')
     
     faces = face_classifier.detectMultiScale(frame, scaleFactor=1.2, minSize=(100, 100), maxSize=(400, 400))
     for (x, y, w, h) in faces:
         eyes = eye_classifier.detectMultiScale(faceROI)
         overlay(frame, glasses2, pos)
         
     def overlay(img, glasses, pos):
         # 부분 영상 참조. img1: 입력 영상의 부분 영상, img2: 안경 영상의 부분 영상
         img1 = img[sy:ey, sx:ex]   # shape=(h, w, 3)
         img2 = glasses[:, :, 0:3]  # shape=(h, w, 3)
         alpha = 1. - (glasses[:, :, 3] / 255.)  # shape=(h, w)
     
         # BGR 채널별로 두 부분 영상의 가중합
         img1[..., 0] = (img1[..., 0] * alpha + img2[..., 0] * (1. - alpha)).astype(np.uint8)
         img1[..., 1] = (img1[..., 1] * alpha + img2[..., 1] * (1. - alpha)).astype(np.uint8)
         img1[..., 2] = (img1[..., 2] * alpha + img2[..., 2] * (1. - alpha)).astype(np.uint8)
     ```



# 9. 특징점 검출과 매칭







































