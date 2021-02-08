---
layout: post
title: 【CV】Computer Vision at FastCampus2, chap7~10
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

   - 그래프 알고리즘을 이용해서 Segmentation을 수행하는 알고리즘 (정확한 알고리즘은 [논문 참조](https://grabcut.weebly.com/background--algorithm.html))
   - **cv2.grabCut(img, mask, rect)**  
     **mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')**  
     **dst = src * mask2[:, :, np.newaxis]**
   - 마우스를 활용한 그랩컷 영상 분할 예제 : grabcut2.py  크게 어렵지 않음

2. 모멘트 기반 (비슷한 모양 찾기 기법을 이용한) 내가 찾고자 하는 객체 검출

   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210125090819511.png?raw=tru" alt="image-20210125090819511" style="zoom:80%;" />
   - Hu's seven invariant moments : 크기, 회전, 이동, 대칭 변환에 불변
   - 모양 비교 함수: **cv2.matchShapes(contour1, contour2, method, parameter)** -> 영상 사이의 거리(distance)

3. 템플릿 매칭

   - 입력영상에서 작은 크기의 템플릿과 일치하는 부분 찾는 기법

   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210125091301957.png?raw=tru" alt="image-20210125091301957" style="zoom:80%;" />

   - **cv2.matchTemplate(image, templ, method, result=None, mask=None) -> result**   
     image의 크기가 W x H 이고, templ의 크기가 w x h 이면 result 크기는 (W - w + 1) x (H - h +1)

   - method 부분에 들어가야할, distance 구하는 수식은 강의 및 강의 자료 참조

   - ```python
     res = cv2.matchTemplate(src, templ, cv2.TM_CCOEFF_NORMED)
     res_norm = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
     _, maxv, _, maxloc = cv2.minMaxLoc(res)
     ```

4. 템플릿 매칭 (2) - 인쇄체 숫자 인식

   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210125092302957.png?raw=tru" alt="image-20210125092302957" style="zoom:80%;" />
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

   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210125093630644.png?raw=tru" alt="image-20210125093630644" style="zoom: 80%;" />

     - 9개 : 180도를 20도 단위로 나눠서 9개 단위로 gradient 분류
     - 1개 셀 8x8, 1개 블록 16 x 16. 블록 1개 는 36개(4블록 x 9개 Gradient)의 히스토그램 정보를 가짐

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

1. 코너 검출

   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210126092522678.png?raw=tru" alt="image-20210126092522678" style="zoom: 67%;" />
   - **cv2.cornerHarris**(src, blockSize, ksize, k)
   - **cv2.goodFeaturesToTrack**(image, maxCorners, qualityLevel, minDistance)
   - **cv2.FastFeatureDetector_create**(, threshold=None, nonmaxSuppression=None, type=None)  
     **cv2.FastFeatureDetector.detect**(image) -> keypoints
   - 예제 및 사용법은 강의 자료 참조

2. 특징점 검출 (local 영역만의 특징(Discriptor )을 가지는 곳을 특징점 이라고 한다.)

   - SIFT, KAZE, AKAZE, ORB 
   - 아래의 방법들을 사용해서 <u>feature</u> 객체 생성
   - **cv2.KAZE_create**(, ...) -> retval 
   - **cv2.AKAZE_create**(, ...) -> retval 
   - **cv2.ORB_create**(, ...) -> retval 
   - **cv2.xfeatures2d.SIFT_create**(, ...) -> retval
   - <u>feature</u>.**detect**(image, mask=None) -> keypoints
   - **cv2.drawKeypoints(image, keypoints, outImage,** color=None, flags=None) -> outImage

3. 기술자 (Descriptor, feature vector)

   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210126093448937.png?raw=tru" alt="image-20210126093448937" style="zoom:80%;" />
   - 특징점 근방의 Local feature을 표현하는 실수 또는 이진 벡터. 위에서는 하나의 특징점이 64개 원소의 백터를 기술자로 가진다. 
   - 실수 특징 백터. 주로 백터 내부에는 방향 히스토그램을 특징 백터로 저장하는 알고리즘 : SIFT, SURF, KAZE
   - Binary descriptor. 주변 픽셀값 크기 테스트 값을 바이너리 값으로 저장하는 알고리즘 : AKAZE, ORB, BRIEF
   - 위 2. 특징점 검출에서 만든 <u>feature 객체</u>를 사용
     - **cv2.Feature2D.compute(image, keypoints)** -> keypoints, descriptors (이미 keypoint 있다면)
     - **cv2.Feature2D.detectAndCompute(image)** -> keypoints, descriptors
   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210126094008801.png?raw=tru" alt="image-20210126094008801" style="zoom: 60%;" />
   - KAZE, AKAZE이 속도 면에서 괜찮은 알고리즘. SIFT가 성능면에서 가장 좋은 알고리즘

4. 특징점 매칭 (feature point matching)

   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210126094928068.png?raw=tru" alt="image-20210126094928068" style="zoom: 60%;" />
   - <u>matcher</u> 객체 생성 : **cv2.BFMatcher_create(, normType=None, crossCheck=None)**  
   - matching 함수1 : <u>matcher</u>.**match(queryDescriptors, trainDescriptors)**
   - matching 함수2 : <u>matcher</u>.**knnmatch(queryDescriptors, trainDescriptors)**
   - **cv2.drawMatches(img1, keypoints1, img2, keypoints2)**

5. 좋은 매칭 선별

   - 가장 좋은 매칭 결과에서 distance 값이 작은 것부터 사용하기 위해,

   - **cv2.DMatch.distance** 값을 기준으로 정렬 후 상위 N개 선택

   - ```python
     # 특징점 매칭
     matcher = cv2.BFMatcher_create()
     matches = matcher.match(desc1, desc2)
     
     # 좋은 매칭 결과 선별 1번 (선발되는 mathcing 수는 내가 선택하기 나름)
     matches = sorted(matches, key=lambda x: x.distance)
     good_matches = matches[:80]
     # 좋은 매칭 결과 선별 2번 (전체 매칭 3159개 중, 384개가 선발됨)
     good_matches = []
     for m in matches:
     if m[0].distance / m[1].distance < 0.7:
     good_matches.append(m[0])
     
     # 특징점 매칭 결과 영상 생성
     dst = cv2.drawMatches(src1, kp1, src2, kp2, good_matches, None)
     ```

6. 호모그래피와 영상 매칭

   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210126100307371.png?raw=tru" alt="image-20210126100307371" style="zoom: 60%;" />

   - **cv2.findHomography(srcPoints, dstPoints) -> retval, mask**

   - good_matches에서 queryIdx, trainIdx 와 같이 2장의 이미지 각각에 대한 특징점 검출 됨.

   - pts1 = np.array([kp1[m.queryIdx].pt **for** m in good_matches] ).reshape(-1, 1, 2).astype(np.float32) pts2 = np.array([kp2[m.trainIdx].pt **for** m in good_matches] ).reshape(-1, 1, 2).astype(np.float32)

   - H, _ = **cv2.findHomography**(pts1, pts2, cv2.RANSAC)

   - ```python
     # 좋은 매칭 결과 선별
     matches = sorted(matches, key=lambda x: x.distance)
     good_matches = matches[:80]
     # 호모그래피 계산
     pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches]
     ).reshape(-1, 1, 2).astype(np.float32)
     pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches]
     ).reshape(-1, 1, 2).astype(np.float32)
     # Find Homography
     H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
     # 일단 matching된거 그리기
     dst = cv2.drawMatches(src1, kp1, src2, kp2, good_matches, None,
     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
     # perspectiveTransform 하기 위한 다각형 꼭지점 설정
     (h, w) = src1.shape[:2]
     corners1 = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]
     ).reshape(-1, 1, 2).astype(np.float32)
     # perspectiveTransform 적용
     corners2 = cv2.perspectiveTransform(corners1, H)
     corners2 = corners2 + np.float32([w, 0]) # drawMatches에서 오른쪽 영상이 왼쪽 영상 옆에 붙어서 나타나므로, 오른쪽 영상을 위한 coners2를 그쪽까지 밀어 줘야 함
     # 다각형 그리기
     cv2.polylines(dst, [np.int32(corners2)], True, (0, 255, 0), 2, cv2.LINE_AA)
     ```

   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210126101521138.png?raw=tru" alt="image-20210126101521138" style="zoom:80%;" />

7.  이미지 스티칭

   - 동일 장면의 사진을 자연스럽게(seamless) 붙여서 한 장의 사진으로 만드는 기술

   - **특징점과 matching** 등등 매우 복잡한 작업이 필요하지만, OpenCV에서 **하나의 함수**로 구현되어 있다.

   - **cv2.Stitcher_create(, mode=None) -> retval, pano**

   - ```python
     # 이미지 가져오기
     img_names = ['img1.jpg', 'img2.jpg', 'img3.jpg']
     imgs = []
     for name in img_names:
     img = cv2.imread(name)
     imgs.append(img)
     # 가져온 이미지, Stitcher에 때려넣기
     stitcher = cv2.Stitcher_create()
     _, dst = stitcher.stitch(imgs)
     cv2.imwrite('output.jpg', dst)
     ```

8. : AR 비디오 플레이어

   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210126101754082.png?raw=tru" alt="image-20210126101754082" style="zoom: 80%;" />

   - 아래의 코드는 핵심만 기술해 놓은 코드. 전체는 ARPlayer.py파일 참조

   - ```python
     # AKAZE 특징점 알고리즘 객체 생성
     detector = cv2.AKAZE_create()
     # 기준 영상에서 특징점 검출 및 기술자 생성
     kp1, desc1 = detector.detectAndCompute(src, None)
     # 해밍 거리를 사용하는 매칭 객체 생성
     matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
     while True:
         ret1, frame1 = cap1.read() # 카메라 영상(Reference Image 나옴)
         # 호모그래피 계산
         H, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC)
         # 비디오 프레임을 투시 변환
         video_warp = cv2.warpPerspective(frame2, H, (w, h))
     
         white = np.full(frame2.shape[:2], 255, np.uint8) # Video 파일
         white = cv2.warpPerspective(white, H, (w, h))
     
         # 비디오 프레임을 카메라 프레임에 합성
         cv2.copyTo(video_warp, white, frame1)
     ```



# 10. 객체 추적과 모션 백터

1. <u>배경 차분 : 정적 배경 차분</u>

   - 배경 차분(Background Subtraction: BS) : 등록된 배경 이미지과 현재 입력 프레임 이미지와의 차이(img-src) 영상+Threshold을 이용하여 전경 객체를 검출
   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210128090501840.png?raw=tru" alt="image-20210128090501840" style="zoom: 80%;" />
   - 위의 Foreground mask에다가, 가이시안 필터 -> 레이블링 수행 -> 픽셀 수 100개 이상은 객체만 바운딩 박스 표시
   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210128090747725.png?raw=tru" alt="image-20210128090747725" style="zoom:80%;" />

2. <u>배경 차분 : 이동 평균 배경</u>

   - 위의 방법은, 조도변화에 약하고 주차된 차도 움직이지 않아야 할 민큼 배경 이미지가 불변해야 한다.
   - 이와 같은 평균 영상을 찾자  
     ![image-20210128091052381](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210128091052381.png?raw=tru)
   - 매 프레임이 들어올 때마다 평균 영상을 갱신   
      <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210128091120398.png?raw=tru" alt="image-20210128091120398" style="zoom: 67%;" />
   - **cv2.accumulateWeighted**(src, dst, alpha, mask=None) -> dst  
     즉, dst(x ,y ) = (1 - alpha) * dst(x ,y ) + alpha src(x ,y ) 

3. <u>배경 차분 : MOG 배경 모델(Mixture of Gaussian = Gaussian Mixture Model))</u>

   - 배경 픽셀값 하나하나가, 어떤 가오시간 분표를 따른다고 정의하고 그 분포를 사용하겠다.

   - **각 픽셀**에 대해 MOG 확률 모델을 설정하여 배경과 전경을 구분 (구체적인 내용은 직접 찾아서 공부해보기-paper : Improved adaptive Gaussian mixture model for background subtraction) 

   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210128100755318.png?raw=tru" alt="image-20210128100755318" style="zoom:80%;" />

   - ```python
     cap = cv2.VideoCapture('PETS2000.avi')
     bs = cv2.createBackgroundSubtractorMOG2()
     #bs = cv2.createBackgroundSubtractorKNN() # 직접 테스트 하고 사용 해야함. 뭐가 더 좋다고 말 못함.
     #bs.setDetectShadows(False) # 125 그림자값 사용 안함
     
     while True:
         ret, frame = cap.read()
         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         fgmask = bs.apply(gray)  # 0(검) 125(그림자) 255(백)
         back = bs.getBackgroundImage()
     ```

   - 동영상을 보면, **생각보다 엄청 잘되고, 엄청 신기하다...**

4. <u>평균 이동(Mean shift) 알고리즘</u> 

   - Tracking : **Mean Shift, CamShift, Optical Flow, Trackers in OpenCV 3.x**

   - Mean shift=mode seeking : 데이터가 가장 밀집되어 있는 부분을 찾아내기 위한 방법, 예를 들어 가오시안 이면 평균 위치를 찾는 방법. 아래에 하늘색 원을 랜덤으로 생성한 후, 그 내부의 빨간색 원들의 x,y평균을 찾는다. 그리고 그 x,y평균점으로 하늘색 원을 옮겨 놓는다(이래서 Mean shift). 이 작업을 반복하다 보면, 결국 하늘색 원은 빨간색 원이 가장 밀집한 곳으로 옮겨 가게 된다.   
     <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210128101804262.png?raw=tru" alt="image-20210128101804262" style="zoom:80%;" />

   - 사람의 얼굴 살색을, [히스토그램 역투영법](https://junha1125.github.io/blog/self-study/2021-01-13-fast_campus1/#chap3---%EA%B8%B0%EB%B3%B8-%EC%98%81%EC%83%81-%EC%B2%98%EB%A6%AC-%EA%B8%B0%EB%B2%95)으로 찾은 후 그 영역에 대한 평균점을 찾아가면서 Tracking을 한다. 

   - **cv2.meanShift(probImage, window, criteria) -> retval, window**

     - probImage : 히스토그램 역투영 영생
     - window : 초기 검색 영역 윈도우 & 결과 영역 반환

   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210128102346633.png?raw=tru" alt="image-20210128102346633" style="zoom:80%;" />

   - ```python
     # 첫번째 프레임의 HS 히스토그램 구하기
     hist = cv2.calcHist([roi_hsv], channels, None, [45, 64], ranges)
     # 히스토그램 역투영 & 평균 이동 추적
     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     backproj = cv2.calcBackProject([hsv], channels, hist, ranges, 1)
     _, rc = cv2.meanShift(backproj, rc, term_crit) 
     ```

   - 히스토그램 역투영법, HS 히스토그램에 대해 궁금하면, 동영상 직접 찾아서 보기

   - 단점 : 객체가 항상 같은 크기이여야 함. 예를 들어, 위의 귤이 멀어져서 작아지면 검출 안된다.

5. <u>[Cam Sift(캠시프트)](https://fr.wikipedia.org/wiki/Camshift) 알고리즘</u> 

   - 위의 단점을 해결하기 위한 알고리즘, 위의 평균 이동 알고리즘을 이용. 

   - 일단 평균 이동을 통해 박스를 이용한다. 그리고 히스토그램 역투영으로 나오는 영역에 대해서, 최적의 타원을 그린다. 만약 타원이 평균이동박스 보다 작으면, 이동박스를 작게 변경한다. 반대로 최적의 타원이 박스보다 크다면, 이동박스를 크게 변경한다. 이 과정을 반복한다. 

   - **cv2.CamShift**(probImage, window, criteria) -> retval, window

   - ```python
     # HS 히스토그램에 대한 역투영 & CamShift
     frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     backproj = cv2.calcBackProject([frame_hsv], channels, hist, ranges, 1)
     ret, rc = cv2.CamShift(backproj, rc, term_crit)
     ```

6. <u>루카스-카나데 옴티컬 플로우(OneDrive\20.2학기\컴퓨터비전\OpticalFlow.pdf참조)</u>

   - Optical flow : 객체의 움직임에 의해 나타나는 객체의 이동 (백터) 정보 패턴. 아래 식에서 V는 객체의 x,y방향 움직임 속도이고, I에 대한 미분값은 엣지검출시 사용하는 픽셀 미분값이다. (컴퓨터비전-윤성의교수님 강의 자료에 예시 문제 참고)    	
     <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210129120210714.png?raw=tru" alt="image-20210129120210714" style="zoom:80%;" />

   - 추가 가정 : 이웃 픽셀은 같은 Flow를 가짐 → NxN Window를 사용하면 N^2개 방정식 → Least squares method

   - 루카스-카나데 알고리즘(Lucas-Kanade algorithm)  

     - cv2.**calcOpticalFlowPyrLK**(…) : input parameter는 강의자료 + Official document 공부
     - **Sparse points**에 대한 이동 벡터 계산 → **특정 픽셀**에서 옵티컬플로우 벡터 계산
     - 몇몇 특정한 점들에 대해서만, Optical Flow를 계산하는 방법

   -  파네백 알고리즘(Farneback's algorithm)

     - cv2.**calcOpticalFlowFarneback**(…) : input parameter는 강의자료 + Official document 공부
     - **Dense points**에 대한 이동 벡터 계산 → **모든 픽셀**에서 옵티컬플로우 벡터 계산
     - 이미지 전체 점들에 대해서, Optical Flow를 계산하는 방법

   - ```python
     # 루카스-카나데 알고리즘(Lucas-Kanade algorithm)  
     pt1 = cv2.goodFeaturesToTrack(gray1, 50, 0.01, 10)
     pt2, status, err = cv2.calcOpticalFlowPyrLK(src1, src2, pt1, None)
     # 2개 이미지 겹친 이미지 만들기
     dst = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)
     # 화면에 백터 표현하기
     for i in range(pt2.shape[0]):
         if status[i, 0] == 0:
             continue
         cv2.circle(dst, tuple(pt1[i, 0]), 4, (0, 255, 255), 2, cv2.LINE_AA)
         cv2.circle(dst, tuple(pt2[i, 0]), 4, (0, 0, 255), 2, cv2.LINE_AA)
         cv2.arrowedLine(dst, tuple(pt1[i, 0]), tuple(pt2[i, 0]), (0, 255, 0), 2)
   ```
     
   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210129120901079.png?raw=tru" alt="image-20210129120901079" style="zoom:80%;" />

7. <u>밀집 옵티컬플로우 (파네백 알고리즘 예제)</u>

   - 만약 필요하다면, 아래의 코드를 그대로 가져와서 사용하기. 한줄한줄 이해는 (강의자료 보는것 보다는) 직접 찾아보고 ipynb에서 직접 쳐봐서 알아내기, 어렵지 않음

   - ```python
     # dense_op1.py 
     flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,0.5, 3, 13, 3, 5, 1.1, 0)
     
     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
     hsv[..., 0] = ang*180/np.pi/2
     hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
     
     bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
     
     cv2.imshow('frame', frame2)
     cv2.imshow('flow', bgr)
     
     gray1 = gray2
     ```

   - ```python
     # # dense_op2.py
     def draw_flow(img, flow : calcOpticalFlowFarneback의 out값, step=16):
         h, w = img.shape[:2]
         y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
         fx, fy = flow[y, x].T
         lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
         lines = np.int32(lines + 0.5)
         vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
         cv2.polylines(vis, lines, 0, (0, 255, 255), lineType=cv2.LINE_AA)
     
         for (x1, y1), (_x2, _y2) in lines:
             cv2.circle(vis, (x1, y1), 1, (0, 128, 255), -1, lineType=cv2.LINE_AA)
     
         return vis
     ```

   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210129121417606.png?raw=tru" alt="image-20210129121417606" style="zoom:80%;" />

   - Optical flow를 사용하기 위해서 추천하는 함수들 : 맨 위가 가장 parents,super class이고 아래로 갈 수록 상속을 받는 Derived class,child class,sub class 등이 있다.

     - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210129121815328.png?raw=tru" alt="image-20210129121815328" style="zoom:80%;" />

8. <u>OpenCV 트래커</u>

   - OpenCV 4.5 기준으로 4가지 트래킹 알고리즘 지원 (4.1 기준 8가지 지원. 사용안되는건 지원에서 빼 버린것 같다)

   - **TrackerCSRT, TrackerGOTURN, TrackerKCF, TrackerMIL**

   - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210129122329568.png?raw=tru" alt="image-20210129122329568" style="zoom: 67%;" />

   - ```python
     cap = cv2.VideoCapture('tracking1.mp4')
     tracker = cv2.TrackerKCF_create()
     ret, frame = cap.read()
     rc = cv2.selectROI('frame', frame)
     tracker.init(frame, rc)
     while True:
         ret, frame = cap.read()
         ret, rc = tracker.update(frame)
         rc = [int(_) for _ in rc]
         cv2.rectangle(frame, tuple(rc), (0, 0, 255), 2)
     ```

   - ![image-20210129122403908](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210129122403908.png?raw=tru)

9. 실전 코딩: 핸드 모션 리모컨

   - 움직임이 있는 영역 검출 / 움직임 벡터의 평균 방향 검출

   - **cv2.calcOpticalFlowFarneback()** -> 움직임 벡터 크기가 특정 임계값(e.g. 2 pixels)보다 큰 영역 안의 움직임만 고려

   -  움직임 벡터의 x방향 성분과 y방향 성분의 평균 계산

     - ```python
       mx = cv2.mean(vx, mask=motion_mask)[0]
       my = cv2.mean(vy, mask=motion_mask)[0]
       m_mag = math.sqrt(mx*mx + my*my)
       
       if m_mag > 4.0:
           m_ang = math.atan2(my, mx) * 180 / math.pi
           m_ang += 180
       ```

   - FastCampus_CV\opencv_python_ch06_ch10\ch10\hand_remocon.py

