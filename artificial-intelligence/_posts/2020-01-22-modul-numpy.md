---
layout: post
title: (모듈) Numpy Scipy Matplotlib 기초
# description: > 
    
---
 

원본 페이지 : https://junha1125.tistory.com/41?category=835551



[CS231 [Justin Johnson\] 교수님이 작성해주신 글로 공부한 내용입니다.](http://cs231n.github.io/python-numpy-tutorial/)**

[ Python Numpy TutorialThis tutorial was contributed by Justin Johnson. We will use the Python programming language for all assignments in this course. Python is a great general-purpose programming language on its own, but with the help of a few popular libraries (numpy, scipy,cs231n.github.io](http://cs231n.github.io/python-numpy-tutorial/)



# **1. Numpy**

\- [내용 중간중간에 있는 링크가 매우 유용하니 참고할 것]

\- 배열은 동일한 자료형을 가지는 값들

\- 값들은 튜플 형태로 색인 된다. 

\- rank : 몇차원 행렬인가?

\- shape : 몇 곱하기 몇 행렬인가. channel * hight * width

배열 생성

> \>> a = np.array( [[1, 2, 2], [2, 2, 2]] )
> \>> a.shape

배열 생성 함수

> a = np.zeros([2,2]) # 2*2 0행렬 생성
> b = np.ones((1,2)) # ()를 사용하던 []를 사용하던 상관 x
> c = np.full((2,2), 7) # 모든 값을 7로 채운 배열
> d = np.eye(2)  # 단위백터
> e = np.random.random((2,2)) # random vector 생성

numpy배열 인덱싱하기. (슬라이싱)

> a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
> a[1] == [5,6,7,8]
> a[1][3] == 7
> a[:2, 1:3] == 0,1행과 1,2열 == [[2 ,3],[6,7]]
> b = a[:2, 1:3] # 단순복사 -> copy of reference

numpy배열 인덱싱하기. (불연속적인 원소 가져오기)

> a = np.array([[1,2], [3, 4], [5, 6]])
> b = a[[0, 1, 2], [0, 1, 0]] == [a[0, 0], a[1, 1], a[2, 0]]  # 즉 행렬의 (0,0) (1,1) (2,0) 원소만 가져오기 즉 1*3행렬
>
> **응용하기.**
> b = np.array([0, 2, 0, 1])
> c = a[np.arange(4), b] # 즉 a행렬의 (0,0) (1,2) (2,0) (3 1)원소 즉 1*4행렬 # [arange](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html)/[linespace](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html)
> c += 10
> print(a) # copy of reference이므로 위의 a행렬의 일부 원소도 10이 더해져 있을 것이다. 

불리언 배열(mask 배열)

> a = np.array([[1,2], [3, 4], [5, 6]])
> bool_idx = (a > 2)
> print(a[bool_idx]) # 출력 "[3 4 5 6]"
> print(a[a > 2]) # 출력 "[3 4 5 6]"

자료형

> 배열 원소의 모든 자료형은 동일
> x = np.array([1, 2]) 
> print(x.**dtype**)  # 자료형 알려주는 맴버함수
> x = np.array([1, 2], **dtype=np.int64**) # 특정 자료형을 명시적으로 지정해주는 방법

배열 연산

> x + y // x - y // x * y // x / y # 기본 사칙 연산자를 사용해서 해당 위치의 원소들과 계산 된다.
> np.sqrt(x)  # 배열 전체에 sqrt처리 하기
>
> 행렬의 곱을 위해서는 dot함수를 사용한다. 
> x.dot(y)
> np.dot(x, y)

sum함수[[수학 함수를 다루는 문서](https://docs.scipy.org/doc/numpy/reference/routines.math.html)]

> np.sum(x)  # 행과 열 상관없이 모든 요소의 합
> np.sum(x, axis=0) # 열이 같은 요소들을 합해서 1차원 백터로 return #주의#
> np.sum(x, axis=1) # 행이 같은 요소들을 합해서 1차원 백터로 return #주의#

전치[[배열을 다루는 문서](https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html)]

> x**.T**
> print(x) # 전치가 적용된 x가 출력된다. 

브로드캐스팅

\- shape가 다른 배열 간에도 산술 연산이 가능하게 하는 메커니즘

\- 예를 들어, 행렬의 각 행에 상수 벡터를 더하는 것

> x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])   # 4*3 행렬
> v = np.array([1, 0, 1])                         # 1*3 행렬
> y = x + v                     # v라는 행렬이 행 방향으로 자동 확대 된다.

[broadcasting되는 조건] 은 아래와 같다.

1. 두 배열이 동일한 rank를 가지고 있지 않다면, 낮은 rank의 1차원 배열이 높은 rank 배열의 shape로 간주합니다.
2. 특정 차원에서 두 배열이 동일한 크기를 갖거나, 두 배열 중 하나의 크기가 1이라면 그 두 배열은 특정 차원에서 compatible하다고 여겨집니다.
3. 두 행렬이 모든 차원에서 compatible하다면, 브로드캐스팅이 가능합니다.
4. 브로드캐스팅이 이뤄지면, 각 배열 shape의 요소별 최소공배수로 이루어진 shape가 두 배열의 shape로 간주합니다.
5. 차원에 상관없이 크기가 1인 배열과 1보다 큰 배열이 있을 때, 크기가 1인 배열은 자신의 차원 수만큼 복사되어 쌓인 것처럼 간주합니다.

[추가]

> np.reshape(v, (3, 1))  # 1*3행렬의 v가 3*1행렬이 된다. reshape를 하려면, 총 요소의 갯수가 같아야 한다.

[broadcasting을 활용한 화려한 연산] :브로드캐스팅은 보통 코드를 간결하고 빠르게 해준다. 따라서 가능한 많이 사용해야 한다. 

```python
import numpy as np

###   1.   ### 
# 벡터의 외적을 계산
v = np.array([1,2,3])  # v의 shape는 (3,)
w = np.array([4,5])    # w의 shape는 (2,)
# 외적을 계산하기 위해, 먼저 v를 shape가 (3,1)인 행벡터로 바꿔야 합니다;
# 그다음 이것을 w에 맞춰 브로드캐스팅한뒤 결과물로 shape가 (3,2)인 행렬을 얻습니다,
# 이 행렬은 v와 w 외적의 결과입니다:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print np.reshape(v, (3, 1)) * w

###   2.   ###
# 벡터를 행렬의 각 행에 더하기
x = np.array([[1,2,3], [4,5,6]])
# x는 shape가 (2, 3)이고 v는 shape가 (3,)이므로 이 둘을 브로드캐스팅하면 shape가 (2, 3)인
# 아래와 같은 행렬이 나옵니다:
# [[2 4 6]
#  [5 7 9]]
print x + v

###   3.   ###
# 벡터를 행렬의 각 행에 더하기
# x는 shape가 (2, 3)이고 w는 shape가 (2,)입니다.
# x의 전치행렬은 shape가 (3,2)이며 이는 w와 브로드캐스팅이 가능하고 결과로 shape가 (3,2)인 행렬이 생깁니다;
# 이 행렬을 전치하면 shape가 (2,3)인 행렬이 나오며
# 이는 행렬 x의 각 열에 벡터 w을 더한 결과와 동일합니다.
# 아래의 행렬입니다:
# [[ 5  6  7]
#  [ 9 10 11]]
print (x.T + w).T
# 다른 방법은 w를 shape가 (2,1)인 열벡터로 변환하는 것입니다;
# 그런 다음 이를 바로 x에 브로드캐스팅해 더하면
# 동일한 결과가 나옵니다.
print x + np.reshape(w, (2, 1))

###   4.   ###
# 행렬의 스칼라배:
# x 의 shape는 (2, 3)입니다. Numpy는 스칼라를 shape가 ()인 배열로 취급합니다;
# 그렇기에 스칼라 값은 (2,3) shape로 브로드캐스트 될 수 있고,
# 아래와 같은 결과를 만들어 냅니다:
# [[ 2  4  6]
#  [ 8 10 12]]
print x * 2
```



------

# **2. SciPy**

numpy를 바탕으로 만들어진 라이브러리. numpy보다 더 많은 함수를 제공하고 과학, 공학분야에 사용된다.

**a. 이미지 작업하기**

> from scipy.misc import imread, imsave, imresize
> img = imread('assets/cat.jpg')                          # 이미지 가져오기
> img.dtype, img.shape                                 # 이미지의 shape와 type알아보기
> img_tinted = img * [1, 0.95, 0.9]                 # broadcasting을 이용해서 RGB각각에 1 0.95 0.9 곱해주기
> img_tinted = imresize(img_tinted, (300, 300))              # 이미지 크기 바꾸기 (400*248) -> (300*300)
> imsave('assets/cat_tinted.jpg', img_tinted)                 # 바꾼 이미지 저장하기

**b. matlab 파일 다루기**

scipy.io.loadmat 와 scipy.io.savemat함수를 통해 matlab 파일을 읽고 쓸 수 있다.

**c. 두 점 사이의 거리**

scipy.spatial.distance.pdist함수는 주어진 점들 사이의 모든 거리를 계산한다. 혹은 scipy.spatial.distance.cdist

> import numpy as np from scipy.spatial.distance import **pdist,** **squareform**
> d = squareform(pdist(x, 'euclidean'))

 

------

# 3. Matplotlib

**a. 함수그리기**

matplotlib.pyplot 모듈 내부에 있는 함수를 이용하면 편하다. 

> import matplotlib.pyplot as plt
> x = np.**arange**(0, 3 * np.pi, 0.1)
> y = np.sin(x)
> plt.subplot(2, 1, 1) # 2*1 행렬에 1번째 원소로 아래의 그래프가 들어간다. 
> plt.plot(x, y)
> plt.xlabel('x axis label')
> plt.ylabel('y axis label')
> plt.title('Sine and Cosine')
> plt.legend(['Sine', 'Cosine'])
> plt.show()

**b. 이미지 나타내기**

> from scipy.misc import imread, imresize
> import matplotlib.pyplot as plt
> \# 이미지 가져오기 scipy를 이용해서 가져오기
> img = imread('assets/cat.jpg')
> \# 1
> plt.subplot(1, 2, 1)
> plt.imshow(img)
> \# 2
> plt.subplot(1, 2, 2)
> plt.imshow(np.uint8(img_tinted))
>
>
> 
> plt.show()