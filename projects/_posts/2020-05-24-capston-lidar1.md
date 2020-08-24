---
layout: post
title: 【캡스톤1】 rpLidar + 라즈베리파이 기본 설정하기 1
description: >  
    캡스톤 수업을 위한 rpLidar + 라즈베리파이 기본 설정하기 1

---

 캡스톤 수업을 위한 rpLidar + 라즈베리파이 기본 설정하기 1

# 1. REFERENCE

1. [로스 패키지 구동 네이버 블로그](https://m.blog.naver.com/PostView.nhn?blogId=thumbdown&logNo=220385363246&proxyReferer=https:%2F%2Fwww.google.com%2F)

2. [로스란? 아두이노란? 아두이노에 Lidar데이터를 가져오는 코드는 여기에](https://www.dfrobot.com/blog-26.html?gclid=EAIaIQobChMI6Jmv6qTM6QIVJdWWCh2Pqw3eEAAYAyAAEgKFHfD_BwE)

3. ROS 홈페이지에서 rplidar를 검색했을 때 나오는 코드 목록

   <img src="C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20200524194150672.png" alt="image-20200524194150672" style="zoom:50%;" />

4. 라이더 드라이버 로스 패키지 목록 (위의 3개 하나하나)

   1. [rplidar](http://wiki.ros.org/rplidar) (distros = kinetic, melodic)
      - [Slamtec/rplidar_ros](https://github.com/Slamtec/rplidar_ros)

   2. [rplidar driver in python]([http://wiki.ros.org/rplidar%20driver%20in%20python](http://wiki.ros.org/rplidar driver in python)) (distros = indigo, jade)
      - [rplidar_python](https://github.com/DinnerHowe/rplidar_python)

   3. [rplidar_ros](http://wiki.ros.org/action/fullsearch/rplidar_ros?action=fullsearch&context=180&value=linkto%3A"rplidar_ros") - 2016년 버전이라 읽을 글 많이 없음
      - [Slamtec/rplidar_ros](https://github.com/Slamtec/rplidar_ros) - 위와 같이 똑같은 사이트로 유도 됨

5. [RPLIDAR and ROS programming](https://www.seeedstudio.com/blog/2018/11/09/rplidar-and-ros-the-best-way-to-learn-robot-and-slam/) - 블로그 설명

6. [ROS and Hector SLAM](https://ardupilot.org/dev/docs/ros-slam.html) - 블로그 설명

7. [RPLidar_Hector_SLAM](https://github.com/NickL77/RPLidar_Hector_SLAM) - 깃 코드



***

# 2. 읽은 내용 정리하기

Reference 읽기 순서 : 4.1  ->  5  -> 6  ->  7

## (1) Ref 4.1 : [rplidar](http://wiki.ros.org/rplidar)

```
차례
1. Overview
2. ROS Nodes
   - rplidarNode
      - Published Topics
      - Services
      - Parameters
3. Device Settings
4. Launch File Examples
5. About Slamtec/RoboPeak
6. Distributor Links
7. Related Links
```

1. Overview

   rplidar에 대한 기본적 설명. [sensor_msgs/LaserScan](http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html)형식 구조체의 메세지를 publish

   SLAM based on RPLIDAR and ROS Hector Mapping 동영상 속 [ROS Hector](http://wiki.ros.org/hector_slam)

2. ROS Nodes

   - rplidarNode - RPLIDAR 스캔 결과를 로스 메세지 형태로 publish하는 노드

     - Published Topics 

       - 'scan' : 스캔된 데이터의 메세지

     - Services

       - start_motor, stop_motor 모터의 시작과 중지를 관리하는 서비스

     - Parameters

       - ```sh
         0. Parameter이름 (자료형, default 값)
         1. serial_port (string, default: /dev/ttyUSB0)
             serial port name used in your system.
         2. serial_baudrate (int, default: 115200)
             serial port baud rate.
         3. frame_id (string, default: laser_frame)
             frame ID for the device.
         4. inverted (bool, default: false)
             indicated whether the LIDAR is mounted inverted.뒤집혔나
         5. angle_compensate (bool, default: false)
             indicated whether the driver needs do angle compensation. 각도 보상 필요?
         6. scan_mode (string, default: std::string())
             the scan mode of lidar.
         ```

         

3. Device Setting

   - [rplidar tutorial](https://github.com/robopeak/rplidar_ros/wiki) - rplidar-ros/wiki
     - [rplidar basic document](http://bucket.download.slamtec.com/e680b4e2d99c4349c019553820904f28c7e6ec32/LM108_SLAMTEC_rplidarkit_usermaunal_A1M8_v1.0_en.pdf)
     - [rplidar sdk](https://www.slamtec.com/en/Support#rplidar-a-series) - slamtec에서 재공해 줌 - [SDK Git Code](https://github.com/slamtec/rplidar_sdk)과 연결됨
     - 위의 Ref1에서 봤던 내용이 좀더 간략히 나와 있다. 코드 실행 순서..
       <img src="C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20200524204312223.png" alt="image-20200524204312223" style="zoom:50%;" />
     - 이 사진과 같이 라이더를, 자동차에 해야한다. theta와 d가 publish되는 메세지값에 나타나는 듯 하다.
     - remap the USB serial port name 했을 때의 방법도 나온다. 이건 건들지 말자.

4. Launch File Examples

   ```sh
   $ ls -l /dev |grep ttyUSB
   $ sudo chmod 666 /dev/ttyUSB0
   1.
   $ roslaunch rplidar_ros view_rplidar.launch    #for rplidar A1/A2
   2. 
   $ roslaunch rplidar_ros rplidar.launch
   $ rosrun rplidar_ros rplidarNodeClient
   ```

**Ref 4.1.1 [rplidar_ros](https://github.com/Slamtec/rplidar_ros) 내용은 위의 [rplidar tutorial](https://github.com/robopeak/rplidar_ros/wiki) 내용과 동일**



## (2) Ref 5  : [RPLIDAR and ROS programming](https://www.seeedstudio.com/blog/2018/11/09/rplidar-and-ros-the-best-way-to-learn-robot-and-slam/)





