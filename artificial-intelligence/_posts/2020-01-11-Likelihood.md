---
layout: post
title: 【확통】 최대 우도(가능도) 방법 (Maximum Likelihood Method)
# description: > 
    
---
 
공부를 위해 다음을 참고하였다. 

https://bit.ly/2ufTaQe

\- 확률분포가 어떤 분포인지에 따라서 최대 우도가 얼마인지 추정해본다. 

\- 가장 단순한 베르누이 분포(이항 분포)일때, 이 블로그 내용을 추가 설명하자면,

x는 실험을 했을 때 동전의 앞면이 나온 횟수이고, 뮤(μ)는 1회 실험시 동전 앞면이 나올 확률이다. 

https://bit.ly/2vJVDTi

\- 위의 사이트와 다르게 베르누이 분포(이항 분포)일 때 최대 우도에 대해서 생각해 본다.

\- 이 블로그 내용을 맨 아래에 적어 놓을 예정이다.

**오일석 - 기계학습** 

------

# **기계학습책 내용정리**

### **확률변수**

기계 학습이 처리할 데이터는 불확실한 세상에서 발생하므로, 불확실성을 다루는 확률과 통계를 잘 활용해야 한다.

확률 변수([https://namu.wiki/w/%ED%99%95%EB%A5%A0%20%EB%B3%80%EC%88%98](https://namu.wiki/w/확률 변수))



<img src="https://k.kakaocdn.net/dn/mrJfs/btqB6h6Ta76/9WLK9BiTArNidDdXd8XTSk/img.png" alt="img" style="zoom:50%;" />

<img src="https://k.kakaocdn.net/dn/bIMsKS/btqB6XtjJFW/PUQgOZwoPCnTVF9i0tk6R1/img.png" alt="img" style="zoom: 67%;" />

여기서 소문자 x는 아래의 대문자 X이다.



일정한 확률을 갖고 발생하는 [사건](https://namu.wiki/w/사건)(event) 또는 [사상](https://namu.wiki/w/사상)(事象)에 수치가 부여되는 함수. 일반적으로 **대문자** **X**로 나타낸다. 확률변수 X의 구체적인 값에 대해서는 보통 소문자를 사용해서, 예를 들어 X가 p의 확률로 x의 값을 가진다는 것은 P(X = x) = p 등의 확률함수로 표현할 수 있다. (즉 X는 확률변수 : 가능한 모든 경우 x : 그 가능한 모든 경우 중 특정한 경우 하나)

예시 : X : 주사위를 2번 던졌을때 나오는 모든 경우(36가지) -> x : 1다음에 5가 나오는 경우 -> 이때 P(X = x) = 1/36

 

### **머신러닝에서 확률백터와 확률 분포**



<img src="https://k.kakaocdn.net/dn/bxtqK1/btqB8aMjNf3/1RmXxIGfFy8uhzKBPLfbSK/img.png" alt="img" style="zoom:67%;" />



위의 경우처럼 '가능한 모든 경우'가 확률 분포가 된다. 이때 특정한 x가 나올 확률을 찾는다면, 이와 같다.

P(X = x{5,5,10,6}) = 1/156132156

 

### **결합 확률과 독립 사건**

어떤 2가지 상황이 독립 사건이면 다음과 같은 식이 성립한다.

P(x,y) = p(x)*p(y)

하지만 2가지 상황이 결합이면 다음과 같은 식이 성립한다.(결합 : 2가지 사건이 연결, 연관 되어 있는 경우 ex) 내가 국민대학교를 가서 돈까스를 먹을 확률 = 국민대학교를 갈 확률 * (국민대학교에서)돈까스를 먹을 확룰)



<img src="https://k.kakaocdn.net/dn/nyY2r/btqB7OQesAi/JEW0C3r1fmuYbzK21bCLHK/img.png" alt="img" style="zoom:67%;" />



P(x,y) = p(x)*p(y\|x)

P(x,y) = p(y)*p(x\|x)

이것을 묶은게 오른쪽 베이즈 정리이다. 

 

 

 

### **우도(가능도/likelyhood)**



<img src="https://k.kakaocdn.net/dn/DEn9s/btqB5ikVA7E/gD4zzGBgeX3NbXzE4cRkA1/img.png" alt="img" style="zoom:67%;" />



오른쪽 경우에 대해, 다음의 문제를 생각해보자.

"하얀 공이 나왔다는 사실만 알고 어느 병에서 나왔는지 모르는데, 어느 병 인지 추정하라." 

이는 위에서 배운 베이즈 정리를 이용하면 된다.



<img src="https://k.kakaocdn.net/dn/EBYKM/btqB6ira9Ul/46nC2uKv9OiL12Bj1HuIn0/img.png" alt="img" style="zoom:67%;" />

<img src="https://k.kakaocdn.net/dn/QMLJX/btqB7NX6KGi/jp9tHC2jJaKfXjNXAckG11/img.png" alt="img" style="zoom:67%;" />



다른 관점으로 다시 생각해보자. 베이즈 정리는 [사후확률][우도][사전확률]을 이용한 수식이다. 

사후 확률 : 사건 발생 후의 확률

사전 확률 : 사건 x와 무관하게 미리 알 수 있는 확률

우도 : 사후확률을 구하기 위해 사용! 여기서는 이렇게 사용되었지만, 사실 우도는 이와 같이 쓰인다.

\- 우도 추정 = 역 확률 문제

\- p(알고있음x \| 추정해야 할 사건y) = L(y,x) 라고 표현된다. 

그렇다면 우리가 이 우도를 왜 배우고 있는 것일까? 기계학습의 적용에서 공부해보자.

 

### **우도 지도 학습에 적용**

지도 학습은 x와 y를 다음과 같이 적용할 수 있다. y = 클래스, 라벨, 타겟 x = 특징 백터



<img src="https://k.kakaocdn.net/dn/YkbK1/btqB8aFuXXf/XHFsV73EgILUYKkqusW2dk/img.png" alt="img" style="zoom:67%;" />



예를 들어 오른쪽 그림과 같다. 

이 예제에서 p(y\|x)는 이와 같이 해석할 수 있다. feature가 ...일 때 그것이 ,,,꽃일 확률

이 확률들을 이산확률분포라고 가정하고 일일이 구하는 것은 불가능하다. (특징이 ...일때 이 것은 ,,, 꽃일 확률이 몇%이다. 라고 전부 정의하는 것은 어렵다.) 따라서 우리는 베이즈 정리를 이용해서 p(y\|x)값을 구할 수 있다. 

 

 p(x\|y)를 구하기 쉬운가? x는 높은 차원이 될수 있지만, y는 일반적으로 고정된 샘이다, (유한한 클래스 갯수) 따라서 각각의 클래스들은 독립적으로 확률분포를 추정할 수 있고, 그렇게 독립된 분포를 보고 확률을 추정하면 되므로, 훨씬 쉽다고 할 수 있다. 

 

### **최대 우도법**

의 병 문제를 다시 생각해보자. p((1)\|하양)은 우리가 전체적은 그림을 알고 있다고 가정하고 베이즈 정리를 사용해서 계산을 했다. 그렇다면 다음의 경우에는 어떻게 계산할까? 



<img src="https://k.kakaocdn.net/dn/uf2Gf/btqB6j4HRiB/TbmufGf7kYvup9Q6Ci9D1K/img.png" alt="img" style="zoom:67%;" />

<img src="https://k.kakaocdn.net/dn/EPoJT/btqB79Uaxtx/7BDkQ7f0xlUv5auB8NJy00/img.png" alt="img" style="zoom:67%;" />



다시 한번 생각해보면, 데이터 집합 X는 구할 수 있다. q3는 모른다. q3의 추정값을 찾기 위해서 X를 이용하는 방법이 최대우도법이다.

이때 우리는 X ={흰공, 파랑공, 흰공, 파란공} 과 같은 위의 표현을 이와 같이 쓸 수 있다.

X= {x1, x2, x3,x4} 이것은 위의 식의 P(X\|theta)는 다음과 같이 쓸 수 있다



<img src="https://k.kakaocdn.net/dn/cbw88w/btqB6XNFzzV/lgBKqb51j0qKreP8CUg0tK/img.png" alt="img" style="zoom:67%;" />



이것은 확률의 곱이므로 너무 작은 값이 될 수 있다. 따라서 다음과 같이 최대 로그 우도 추정을 사용한다. 



<img src="https://k.kakaocdn.net/dn/Fpqtk/btqB6X1c3iC/AFWd9IqTQdz5cut4ykYjtK/img.png" alt="img" style="zoom:67%;" />





### 기계학습에서의 최대 우도 공식**

**W = argmax P(X\|W)**

임의의 데이터 분포(date set)에 대해서, 그 분포처럼 나오게 하는, 최대로 적절한 신경망의 가중치 W값

------

# **블로그 내용정리**

### **우도와 가능도**

모수 θ에 의해 만들어진 확률변수 X. [모수](https://support.minitab.com/ko-kr/minitab/18/help-and-how-to/statistics/basic-statistics/supporting-topics/data-concepts/what-are-parameters-parameter-estimates-and-sampling-distributions/)는 위에서 q3와 같이, 분포 곡선을 생성하기 위해 확률 분포 함수(PDF)의 입력 값으로 사용되는 모집단 전체를 설명하는 측도입니다. 



![img](https://k.kakaocdn.net/dn/cuh3BL/btqB6Ii0cON/xcBkiDjn1KRkXFbpK3nq50/img.png)

![img](https://k.kakaocdn.net/dn/b1Yj5y/btqB7ccYwrw/JEIpVvT1N3fjFktNE2dmZ0/img.png)



L(θ\|x)는 표본 x에 대해서 모수 θ의 가능도이다. 풀어 쓰자면, 특정 모수를 가지는 모집단에 대해서, 표본 x가 나왔을 때, 그 표본이 나올수 있게 하는 모수 θ의 가능도(가능성-이 모수를 가지는것이 몇%로 정확한 것인가?)를 L(θ|x)라고 한다. 

참고로 위의 \| 는 조건부 확률의 \|가 아니다. 그냥 모수 θ를 가지는 모집단에서 xn이 나올 확률이다. (이해가 안된다면 맨 위에서 부터 차근차근 아래로 내려오자.)

 

(여기서 부터 모수 θ == 모수 μ)



### **최대 가능도 방법**

표본 x가 있을 때, 가장 가능한(적절한) 모수 θ의 확률(분포)를 구하는 것이다. 

> [이항 분포](https://ko.wikipedia.org/wiki/이항_분포) : 속된 *n*번의 독립적 시행에서 각 시행이 확률 p를 가질 때의 이산 확률 분포이다

> [베르누이 분포](https://ko.wikipedia.org/wiki/베르누이_분포) : 매 시행마다 오직 두 가지의 가능한 결과만 일어난다고 할 때, 이러한 실험을 1회 시행하여 일어난 두 가지 결과에 의해 그 값이 각각 0과 1로 결정되는 확률변수 X



![img](https://k.kakaocdn.net/dn/dOeS8p/btqB6idRHOK/Lq1kqBHCg5d5knJiGXM2E1/img.png)



여기서 μ는 위에서 봤던 모수 θ를의미한다. 그리고 x는 0 또는 1이다.(베르누이 분표에서의 0과 1사용)

직접 0과 1을 대입해보면 P(x=0) = 1-μ 그리고P(x=1) = μ가 됨을 확인할 수 있다.



![img](https://k.kakaocdn.net/dn/Vk3tB/btqB6Jh1dLP/t6yK3EiJHptAhYUebkUkkk/img.png)



p(앞2뒷2\|μ)=μ^2(1−μ)^2 = P(x=0) * P(x=0)* P(x=1) *P(x=1)라고 말할 수 있다. 그리고 이것을 그림으로 그리면 다음과 같이 표현 된다.



![img](https://k.kakaocdn.net/dn/Ev5SR/btqB9xUuIWF/ejVMaURycKbb1VSMU6bfr0/img.png)



이것으로써 우리는 P가 최대(0.063)가 되는 μ가 0.5 라는 것을 알 수 있다.

이것을 일반화해서 적으면 다음과 같다. 



![img](https://k.kakaocdn.net/dn/cUpSCl/btqB8P19IB9/hhmctamsCtvdDeKxQhIKi0/img.png)



즉 다시 말해서, 관측된 Data가 나올 확률을 Likelihood라고 하고, 이 방법을 Maximum Likelihood Estimation, 줄여서 MLE라고 한다. 또 위의 파란색 부분을 일반화 하면



<img src="https://k.kakaocdn.net/dn/3CwNa/btqB8Q03Puo/WKbuzzodzm8Vg0AvenTwz1/img.png" alt="img" style="zoom: 80%;" />



자. 위에서 내가 μ에 대해서 그래프를 그렸다. P가 최대가 되는 μ를 찾기 위해 항상 그래프를 그려야할까??

아니다. μ에 대해서 미분을 하면 되지 않는가?? 그리고 미분한 그 함수가 0이 되는 지점이 최대값이나 최솟값이 되는 지점이겠지... 

 

그냥 미분을 하면 너무 힘드므로, 바로 위의 식에 양변에 log를 씌우고 미분을 해서, 미분한 함수가 0이 되는 μ값을 찾아보자.



<img src="https://k.kakaocdn.net/dn/n1OcN/btqB6j4T1rj/dyKOZnUs86sId74Spe5qS0/img.png" alt="img" style="zoom:67%;" />



즉 우리가 원하는 μ의 MLE



이때 N은 총 수행 횟수이다. (앞앞앞뒤뒤 -> N = 5) 

즉 xn(동전의 앞면) = 1 에 대해서 μ는 (1/N) *(N/2) = 1/2 가 될 것이다. (시그마 xn ~= N/2 이기 때문에.)

