---
layout: post
title: (논문) feature pyramid networks for object detection
# description: > 
    
---

이 논문을 읽어봐야겠다는 생각이 든 계기는 다음과 같다.

**1.** 친구가 발표했던 EfficentDet에 나오는 그림과 똑같은 그림이 나왔다. 이 논문과 feature pyramid networks에 대해 공부를 한다면 EfficentDet을 좀 더 자세히 이해할 수 있을거라 생각했다.

**2.** Detectron2를 공부해보던 중 다음과 같은 것을 보았다. C4와 DC5는 이미 알고 있는 ResNet을 사용하기 때문에 생소했던 feature pyramid networks를 공부할 필요성이 느껴졌다.



![img](https://k.kakaocdn.net/dn/BM9Ec/btqCKArKFgQ/rcHI0XtNsPG4a1P6HMGBC1/img.png)



**3.** Object Detection 모델이 잘 정리되어있는 깃에 대표적은 backbone이 다음과 같이 적혀있었다. 3개가 다 뭔지 모르지만... 차근히 공부한다는 마음으로 feature pyramid networks 논문부터 읽어야 겠다는 생각이 들었다.



![img](https://k.kakaocdn.net/dn/bXtge4/btqCOzEBSKx/ixBYaURnxcZyhfRFdkhMx0/img.png)



------

## 논문 내용 정리 PPT

![img](https://k.kakaocdn.net/dn/bHa7lC/btqCUzlhqr1/kggmA94X89GwbuXF4mk6dK/img.jpg)

![img](https://k.kakaocdn.net/dn/cJoztw/btqCU6XuC4q/oe36MeNxyTAfcsZkCUr8Xk/img.jpg)

![img](https://k.kakaocdn.net/dn/bjavHA/btqCXvIOCj6/LVqKKmPEhBZzLkd1VPeJF1/img.jpg)

![img](https://k.kakaocdn.net/dn/TKFqJ/btqCTwWUqkx/4ZOlcTirI8KNBYKq3OE3dK/img.jpg)

![img](https://k.kakaocdn.net/dn/whNQ1/btqCTxnWgsl/wZmE0Eqb1qgStJw4RDc7N1/img.jpg)

![img](https://k.kakaocdn.net/dn/cwrJ5N/btqCUxVoe5B/9Qjds24Qzu0qKwoZ808Be0/img.jpg)

![img](https://k.kakaocdn.net/dn/cxx7ed/btqCXwnqrwb/tMxKP5F8MwgEBw2JOTzRU0/img.jpg)

![img](https://k.kakaocdn.net/dn/cLQAC7/btqCVENekpP/1FGcqnLcpyQWELOrTHw4u0/img.jpg)

![img](https://k.kakaocdn.net/dn/Zjurt/btqCWL58RZ5/OKHkPvTKsMANJ4MDEkFQk0/img.jpg)

![img](https://k.kakaocdn.net/dn/bmnFv3/btqCVD8Eezx/acoYzzLaO7IG91nPMeodtk/img.jpg)

![img](https://k.kakaocdn.net/dn/bqW24o/btqCTwJrKDl/hr4fvY50blMwBD11yYBPfk/img.jpg)
