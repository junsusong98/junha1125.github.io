---
layout: post
title: 【3D-Detect】A Survey on 3D object-detection for self-driving
---

- **논문** : [A Survey on 3D Object Detection Methods for Autonomous Driving Applications](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8621614)
- **분류** : 3D Object Detection for self-driving
- 저자 : Eduardo Arnold, Omar Y. Al-Jarrah, Mehrdad Dianati, Saber Fallah
- **읽는 배경** : 연구 주제 정하기 첫걸음
- **읽으면서 생각할 포인트** : 코드와 함께 최대한 완벽히 이해하기. 이해한 것 정확히 기록해두기.
- **느낀점**  : 



# A Survey on 3D Object Detection for Autonomous Driving

## 1. Conclusion, Abstract, Introduction

- 논문 과정 : sensors들의 장단점, datasets에 대해서 알아보고, (1) monocular (2) point cloud based (3) fusion methods 기반으로 Relative Work를 소개한다.
- 3D object Detection에서 나오는 depth information 정보를 통해서 만이, path planning, collision avoidance 를 정확히 구현할 수 있다. 자율주행을 하려면 (1) identification=classification (2) recognition of road agents positions (e.g., vehicles, pedestrians, cyclists, etc.) (3) velocity and class 들을 정확히 알아야 한다.
- failure in the perception(인식 실패)는 sensors limitations 그리고 environment(domain) variations에 의해서, 또는"objects’ sizes", "close or far away from the object" 등과 같은 요인에 의해서 발생한다. 



## 2. Sensors

- dd