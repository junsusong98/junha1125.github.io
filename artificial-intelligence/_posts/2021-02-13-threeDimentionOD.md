layout: post
title: 【3D-Detect】A Survey on 3D object-detection for self-driving

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



## 2. Sensors and Dataset

- Sensors
  1. 각 센서의 장단점 비교.    
          <img src="C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210214143901740.png" alt="image-20210214143901740" style="zoom:67%;" />
  2. Calibration 추천 논문 : "LiDAR and camera calibration using motion estimated by sensor fusion odometry"
- Dataset
  1. KITTI 
     - Sensor calibration, Annotated 3D bounding box 를 제공한다. 
     - Categrized in easy, moderate, hard (객체 크기, 겹칩, 가려짐의 정도에 따라서). 
     - 거의 대부분 주간 주행. Class unbalance가 심각하다. Predominant orientation 이다. (대부분의 객체가 같은 방향을 바라보고 있다.)
  2. Virtual KITTI
     - 직접 게임 엔진을사용해서 KITTI의 이미지와 똑같은 가상 사진을 만들었다. 
     - 이 사진에서 Lighting, 날씨 조건, 차의 색 등을 바꿀 수 있다. 
     - 두개 데이터에 대해서 Transferability를 측정해본 결과, 학습과 테스트를 서로에게 적용했을때 성능 감소가 크게 있지 않았다. 
     - Virtual KITTI로 학습시키고, KITTI로 fine-tunning을 해서 나온 모델로 test해서 가장 좋은 성능을 얻었다. 
  3. Simulation tool
     - game-engines GTA5 (SqueezeSeg  [34])
     - Multi Object Tracking Simulator : “Virtual worlds as proxy for multi-object tracking analysis,”
     - autonomous driving simulator : CARLA-"CARLA: An open urban driving simulator-2017", Sim4CV-"Sim4CV: A photo-realistic simulator for computer vision applications-2018"



## 3. 3D Object Detection Methods

- 아래의 전체 분류 Table 참조

![image-20210214171654498](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210214171654498.png)



## 3-A Monocular Image Based Methods

- 우선 아래의 Table 참조
- 대부분의 탐지 방식 : Region Proposal(2D) -> 3D model matching(3D detection) -> Reprojection to obtain 3D bounding boxes
- **문제점**(이것을 해결하기 위해 아이디어를 계속 생각해 봐야겠다) : 이미지에 Depth 단서가 부족해서, 먼 객체, 멀어서 작은 객체, 겹치는 객체, 잘리는 객체 탐지에 힘듬. Lighting과 Weather condition에 매우 제한적임. 오직 Front facing camera만을 사용해서 얻는 정보가 제한적이다. (Lidar와 달리) 

![image-20210214171814828](C:\Users\sb020\OneDrive\바탕 화면\image-20210214171814828.png)

1. 3DOP [43, 2015] : 원본 이미지에서 3D 정보를 추출하는 가장 기본적인 방법. 필기에서는 이 방법을 0번이라고 칭함.
2. Mono3D [39, 2016] : 3DOP를 사용한다.  KeyPoint-"simple region proposal algorithm using context, semantics, handengineered shape features and location priors."
3. Pham and Jeon [44, 2017] :  Keypoint-"class-independent proposals, then re-ranks the proposals using both monocular images and depth maps." 위의 1번 2번보다 좋은 성능 얻음.
4. 3D Voxel Pattern = 3DVP [41, 2015] : 3D detection의 가장 큰 Challenge는 severe occlusion(객체의 겹침, 잘림이다) 이것을 문제점을 해결하기 위한 논문.  객체가 가려진 부분을 recover하기 위해 노력했다. 2D detect결과에 3D BB결과를 project할 때의 the reprojection error를 최소화함으로써 성능을 높였다. 
5.  SubCNN [45, 2017] : 3DVP의 저자가 만든 방법으로, Class를 예측하는 RPN을 사용한다. 그리고 3DVP를 사용해서 똑같은 3D OD를 거친다.([45]논문에 의하면, 2D Image와 Detection결과에서 Camera Parameter를 사용해서 3D 정보를 복원한다고 한다.)
6. Deep MANTA [42, 2017] : a many task network to estimate vehicle position, part localization and shape. 다음의 과정을 순서대로 한다고 한다. 2D bounding regression  -> Localization -> 3D model matching (제한된 vehicle pose의 문제점을 해결하기 위해서)
7. Mousavian et al. [40, 2017] :  standard 2D object detect 결과에다가 추가로 3D orientation (yaw) and bounding box sizes regression (the network prediction Branch)만을 추가해서 3D bounding box결과를 추론한다. orientation을 예측하는데, L2 regression 보다는 a Multi-bin method 라는 것을 제안해서 사용했다고 한다.
8. 360 Panoramic [46, 2018] :  옆, 뒤 객체도 탐지하기 위해서, 360 degrees panoramic image 사용한다. 데이터 셋 부족 문제를 해결하기 위해서 KITTI 데이터를 Transformation한다. 



## 3-B. Point Cloud Based Methods

- 우선 아래 테이블 참조
- Projection 방법은 Bird-eye view를 사용하거나, 이전에 공부했던 SqueezeSeg와 같은 방법으로 3D는 2D로 투영하여 Point Clouds를 사용한다. 이러한 변환 과정에서 정보의 손실이 일어난다. 그리고 trade-off between time complexity and detection performance를 맞추는 것이 중요하다.
- Volumetric methods는 sparse representation(빈곳들이 대부분이다)이므로 비효율적이고 3D-convolutions를 사용해야한다.
- PointNet과 같은 방법은 using a whole scene point cloud as input 이므로, 정보 손실을 줄인다.

![image-20210214202515874](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210214202515874.png)