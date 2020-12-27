---
layout: post
title: 【캡스톤2】 자율주행 RC카 - 장애물 회피 및 곡선 주행
description: >  
    rplidar, IMU, 아두이노, 라즈베리파이를 이용한 자율주행 RC카 제작하기.

---

- rplidar, IMU, 아두이노, 라즈베리파이를 이용한 자율주행 RC카 제작하기.
- 코드는 현재 github에 private repository로 관리 중 입니다. 필요하신 분은 따로 연락주세요.

# 0. 주행 완성 동영상
- 주행영상 1 (아래 이미지를 클릭!)

[![self-RCcar2](https://i.ytimg.com/vi/R8Ti7Q9i_OA/hqdefault.jpg?sqp=-oaymwEZCPYBEIoBSFXyq4qpAwsIARUAAIhCGAFwAQ==&rs=AOn4CLDkyM3i5Z6sePZQz7C1IwJMXba-Rw)](https://www.youtube.com/watch?v=R8Ti7Q9i_OA "self-RCcar2")

- 주행영상 2 (아래 이미지를 클릭!)

[![self-RCcar](https://i.ytimg.com/vi/6XvvjGV5a18/hqdefault.jpg?sqp=-oaymwEZCPYBEIoBSFXyq4qpAwsIARUAAIhCGAFwAQ==&rs=AOn4CLATClCbMVifmGrR_OyHB-RzTN8owg)](https://www.youtube.com/watch?v=6XvvjGV5a18 "self-RCcar")


# 1. 큰그림 간결 손코딩
- 주의사항
   1. 추후에 하나하나의 함수에 대한 손코딩 꼭 하고 코딩하기
   2. 추후에 어떤 식으로 모듈화하고 이용할 건지 정하고 코딩 하기

## To Do list
- PPT 발표준비하기
- 11월 4일 이후에 팀원에게 코드 설명해주기
- 11월 13일 이후에 아래의 내용들 시작하기 그 전까지 쉬자. 힘들어 죽겠다. 
	- IMU ROS에 연동시키고 subscribe하는 line 추가하기
	- IMU값이 오차 없이 거의 일정한지 확인해보기.(정확하다면 두 직선의 기울기의 평균을 사용할 필요 없음)
	- 아래의 손코딩 주행 알고리즘 코딩하기
		1. 2개의 class에 대한 전처리 맴버변수 4개 정의하기. (팀원과 같이 하기)
		2. 주행 알고리즘 맴버변수 정의하기. (팀원과 같이 하기)
	- theta1, theta2 정확한 값 찾아서 적어놓기

## 0. 대표 상수 및 전역 벽수
- theta1 : 첫 직선 기준 각도
- theta2 : 곡선 후 직선 기준 각도
- wall_c_l_count : wall_c_l가 발견된 갯수

## 1. Lidar Point Clouds를 2D 이미지로 변환하기. 
- 이때 중요한 점은 차가 회전해도 이미지는 회전하지 않도록 아래 중 하나 이용.
    1. 따로 이미지 변수를 만든다. 이것은 차가 회전하면 이미지도 회전하는 이미지. 여기서 감지되는 벽의 기울기가 theta1이 되도록, 회전하지 않는 이미지 만들기.
    2. theta1과 현재 IMU에서 나오는 차량 각도의 차이를 반영하기(IMU오차 걱정됨)

## 2. 변환된 이미지를 아래의 처리 순서대로 하기.  
아래의 과정을 통해, 오차 없이 장애물 및 벽이 감지되도록 노력한다. 
1. 가오시안 필터
2. Threshold로 binary 이미지로 변환
3. canny edge 알고리즘 적용
4. hough Transform으로 장애물, 벽 감지
    
## 3. 감지된 직선이 벽 VS 장애물 판단하기
1. 장애물 : 50cm 이상 20cm 이상의 직선 만. 차량 뒤에 있는 장애물은 감지 하지 않기 위해, 장애물의 y축 좌표 중 큰 값이 차 앞에 있을 때 만 고려하기.
2. 벽 : 1m 이상의 직선 만. theta1, theta2와 -5~5도 차이의 기울기를 가지는 직선
- 위에서 탐지한 객체는 다음 중 하나로 labeling을 한다.   
    1. obstacle_r : right
    2. obstacle_l : left
    3. wall_l : theta1
    4. wall_r : theta1
    5. wall_c_r : theta2 curve right
    6. wall_c_l : theta2 curve left
- 한계점 : 
    - 한 lebel에 대해서 여러개의 직선이 검출된다면? 하나만 선택? 평균값 이용?
    - left, right 벽이 잘 검출되지 않는다면? 직선이 겨우겨우 검출된다면? threshold 낮춰서 직선 검출하기. 꼭 필요한게 검출되지 않는다면 그 순간은 속도 0하기.(?)


## 4. 주행 알고리즘 설계하기
- 경우1 : 장애물과의 거리가 충분히 멀 때
    - 점과 직선사이 거리 이용
    - 속도 上
    - 직선 주행 방법
        1. 두 직선과의 거리가 같도록, 두 직선의 기울기의 평균 기울기가 theta1이 되도록 주행하기. (X)  
        2. 두 직선과의 거리가 같도록, 그리고 차량각이 theta1을 가지게 주행하기. (X)
        3. theta1(혹은 두 직선의 기울기의 평균 기울기)를 가지고, 두 직선 중심의 좌표점을 지나는 직선 (혹은 두 직선의 중앙 직선 구하는 방정식 찾기) 위의 한 점을 path point를 설정하고, 차량이 그 point를 향해 주행하도록 설정하기. (V)

- 경우2 : 장애물과의 거리가 가까울 때
    - 점과 직선사이 거리 이용
    - 전면에 있을 때, 측면에 있을 때 모두 가능
    - 속도 下
    - 주행 방법 
        1. 더 큰 벽을 선택한다? NO. 별로 좋은 방법 아닌듯.
        2. obstacle_l, obstacle_r를 선정하는 알고리즘에 두 직선의 중앙에 대해 어디에 있는가를 이용한다.
        3. 그리고 obstacle의 위치 반대편 벽을 따라서 주행한다. 
        4. theta1(혹은 두 직선의 기울기의 평균 기울기)를 가지고, 한 벽에 대해 일정 거리의 좌표점을 지나는 직선 (혹은 일정 거리를 가지는 직선의 방정식 찾기) 위의 한 점을 path point로 설정하고, 차량이 그 point를 향해 주행하도록 설정하기 

- 경우3 : 곡선 주행 하기
    - 가까이 있는 wall_c_l(우회전일 경우)이 일정 count 이상 탐지가 됐다면. 
    - bool 변수를 switch로써 on 하고 다른 함수 실행 시작하기. 
    - wall_c_r이 감지되지 않으면, wall_c_l을 기준으로 경우2 알고리즘을 사용해서 곡선 주행. 
    - wall_c_r이 감지되기 시작하면, 경우1 알고리즘을 사용해서 직선 주행.

- 최종 조향에서! 차가 이미 많이 비틀어진 상태라면 반대방향으로 조향하도록 상수값의 steering 값 return.(경우1,2. 경우3에 대해 다른 상수값 적용)


---
---
# 2. PseudoCode in detail

## 0. To DO List
1. 가상의 line이 그려져 있는 이미지 만들기
1. c++ class 공부하기
1. 그 이미지를 통해서 class 만들고 테스트 해보기

## 1. class의 종류와 맴버변수와 맴버함수
1. class Line 
    - 맴버변수
        1. x1
        1. y1
        1. x2
        1. y2
        1. length - cm
        1. dis_center - cm
        1. slope - degree
        1. label/class - 1(obstacle_l),2(obstacle_r),11(wall_l),12(wall_r),21(wall_c_r),22(wall_c_l)
        1. position - 1(back or left),2(front or right)
    - 맴버함수
        1. lengh 계산하기
        1. dis_center 계산하기
        1. slope 계산하기

2. class Cal_lines
    - 맴버변수
        1. 현재 검출된 line의 갯수
        1. 검출되는 line 최대 18개
        1. lebel 별 갯수 - 6개의 객체
    - 맴버함수
        1. wall 검출하기
        1. obstacle 검출하기
        1. 갯수 변수에 값 저장해 두기
        
        - 경우1 주행알고리즘
        - 경우2 주행알고리즘
        - 경우3 주행알고리즘


# Basic Code - step1
```cpp
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <istream>

using namespace cv;
using namespace std;

// Class Line --------------------------------
class Line{
public:
	int x1,y1,x2,y2;
	float length, dis_center;
	double slope;
	int label;

	Line();
	Line(int _x1, int _y1, int _x2, int _y2);
	float cal_length();
	float cal_center();
	double cal_slope();
};

Line::Line(){
	x1 = 1;
	y1 = 1;
	x2 = 1;
	y2 = 1;
}

Line::Line(int _x1, int _y1, int _x2, int _y2){
	x1 = _x1;
	y1 = _y1;
	x2 = _x2;
	y2 = _y2;
	length = this->cal_length();
	dis_center = this->cal_center();
	slope = this->cal_slope();
}

float Line::cal_length(){
	return 1;
}

float Line::cal_center(){
	return 1;
}

double Line::cal_slope(){
	return 1;
}

// Class Cal_lines --------------------------------
class Cal_lines{
	public:
		int num_lines;
		Line line[18];
		int num_1, num_2, num_11, num_12, num_21, num_22; 

		void append(Line line_t);
		void labeling_wall();
		void labeling_obstacle();
		void howmany_num();
		Cal_lines();
};

Cal_lines::Cal_lines(){
	num_lines = 0;
	num_1 = num_2 = num_11 = num_12 = num_21 = num_22 = 0;
}

void Cal_lines::append(Line line_t){

}

void Cal_lines::labeling_wall(){

}
void Cal_lines::labeling_obstacle(){

}
void Cal_lines::howmany_num(){

}




// Main --------------------------------

int main()
{
	// 컬러 이미지를 저장할 Mat 개체를 생성합니다.

	cv::Mat img(1200, 1200, CV_8UC3, cv::Scalar(0,0,0));
	line(img, Point(450,400), Point(450,750), Scalar(255,255,255),1);
	line(img, Point(750,400), Point(750,750), Scalar(255,255,255),1);
	line(img, Point(630,450), Point(680,450), Scalar(255,255,255),1);
	line(img, Point(630,450), Point(630,420), Scalar(255,255,255),1);
	line(img, Point(450,300), Point(1100,300), Scalar(255,255,255),1);

	Line temp_line(450,400,450,750);
	cout << temp_line.slope << endl;

	Cal_lines Cal;
	cout << Cal.num_lines << endl;
	
	imshow("result", img);
	waitKey(0);
}
```

# Basic Code - step2
```cpp
// 실행하기 전에 꼭 아래의 문구 터미널에 치기!!
// $ export DISPLAY=:0.0
// circle : https://webnautes.tistory.com/1207
// opencv setup : https://webnautes.tistory.com/933
// shortcut : billd : Ctrl + b
// shortcut : excute : Ctrl + r
// 하지만 반드시 실행은 terminal 에서 하도록 해라!! ./opencv
// last sentence : waitKey(0); 필수이다!
// for is going to iterate only if there is keyboard input!


#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <istream>
#include <math.h>
#define PI 3.14159265

using namespace cv;
using namespace std;

// Class Line --------------------------------
class Line{
public:
	int x1,y1,x2,y2;
	float length, dis_center;
	double slope;
	int label;
	int position;

	Line();
	Line(int _x1, int _y1, int _x2, int _y2);
	float cal_length();
	float cal_center();
	double cal_slope();
	int cal_position();
};

Line::Line(){
	x1 = 1;
	y1 = 1;
	x2 = 1;
	y2 = 1;
}

Line::Line(int _x1, int _y1, int _x2, int _y2){
	x1 = _x1;
	y1 = _y1;
	x2 = _x2;
	y2 = _y2;
	length = this->cal_length();
	slope = this->cal_slope();
	dis_center = this->cal_center();
	position = cal_position();
}

float Line::cal_length(){
	return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

float Line::cal_center(){
	// By https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
	int center_x = 600;
	int center_y = 600;
	float dis = abs(((y2-y1)*center_x - (x2-x1)* center_y + x2*y1 - y2*x1) / (this->length));
	return dis;
}

double Line::cal_slope(){
	// return degree {-180 ~ +180}
	return atan2((y2 - y1),(x2 - x1)) * 180 / PI ; 
}

int Line::cal_position(){
	return 1;
}

// Class Cal_lines --------------------------------
class Cal_lines{
	public:
		int num_lines;
		Line line[18];
		int num_1, num_2, num_11, num_12, num_21, num_22; 

		Cal_lines();
		void append(Line line_t);
		void labeling_wall();
		void labeling_obstacle();
		void howmany_num();
};

Cal_lines::Cal_lines(){
	num_lines = 0;
	num_1 = num_2 = num_11 = num_12 = num_21 = num_22 = 0;
}

void Cal_lines::append(Line line_t){
	// Have to change IF condition 85~95...
	// if(line_t.slope == 90 || line_t.slope == 0){
		if(num_lines<18){
			line[num_lines] = line_t;
			num_lines++;	
	}
	// }
}

void Cal_lines::labeling_wall(){

}

void Cal_lines::labeling_obstacle(){

}

void Cal_lines::howmany_num(){

}




// Main --------------------------------
int main()
{
	// 컬러 이미지를 저장할 Mat 개체를 생성합니다.

	cv::Mat img(1200, 1200, CV_8UC3, cv::Scalar(0,0,0));
	int arr[5][4] = 
					{ {450,400,450,750}, //0
				    	{750,400,750,750}, //1
				    	{630,450,680,450}, //2
				    	{700,420,700,700}, //3
				    	{1100,300, 450,300} //4
					};
	
	line(img, Point(arr[0][0],arr[0][1]), Point(arr[0][2],arr[0][3]), Scalar(255,255,255),1);
	line(img, Point(arr[1][0],arr[1][1]), Point(arr[1][2],arr[1][3]), Scalar(255,255,255),1);
	line(img, Point(arr[2][0],arr[2][1]), Point(arr[2][2],arr[2][3]), Scalar(255,255,255),1);
	line(img, Point(arr[3][0],arr[3][1]), Point(arr[3][2],arr[3][3]), Scalar(255,255,255),1);
	line(img, Point(arr[4][0],arr[4][1]), Point(arr[4][2],arr[4][3]), Scalar(255,255,255),1);
	circle(img, Point(600,600),2,Scalar(255,0,0),2);
	

	Cal_lines lines;
	for(int i = 0; i < 5; i++ ){
		Line temp_line(arr[i][0],arr[i][1],arr[i][2],arr[i][3]);
		lines.append(temp_line);
	}

	cout << lines.num_lines << endl;
	cout << lines.line[0].dis_center << endl;



	
	imshow("result", img);
	waitKey(0);
	//waitKey(1);
}
```

# Intermediate Code - step3
```cpp
// 실행하기 전에 꼭 아래의 문구 터미널에 치기!!
// $ export DISPLAY=:0.0
// circle : https://webnautes.tistory.com/1207
// opencv setup : https://webnautes.tistory.com/933
// shortcut : billd : Ctrl + b
// shortcut : excute : Ctrl + r
// 하지만 반드시 실행은 terminal 에서 하도록 해라!! ./line_test
// last sentence : waitKey(0); 필수이다!
// For(including waitkeyy_function) is going to iterate only if there is keyboard input!


#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <istream>
#include <math.h>
#define PI 3.14159265

using namespace cv;
using namespace std;

// 1. Class Line ----------------------------------------------------------------------------------------------------------------------------
class Line{
public:
	int x1,y1,x2,y2;
	float length, dis_center;
	double slope;
	int label; // 1,2,11,12,21,22
	int position;

	Line();
	Line(int _x1, int _y1, int _x2, int _y2);
	float cal_length();
	float cal_center();
	double cal_slope();
	int cal_position();
};

Line::Line(){
	x1 = 1;
	y1 = 1;
	x2 = 1;
	y2 = 1;
}

Line::Line(int _x1, int _y1, int _x2, int _y2){
	x1 = _x1;
	y1 = _y1;
	x2 = _x2;
	y2 = _y2;
	length = this->cal_length();
	slope = this->cal_slope();
	dis_center = this->cal_center();
	position = this->cal_position();
}

float Line::cal_length(){
	return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

float Line::cal_center(){
	int center_x = 600;
	int center_y = 600;
	float dis = abs(((y2-y1)*center_x - (x2-x1)* center_y + x2*y1 - y2*x1) / (this->length));
	return dis;
}

double Line::cal_slope(){
	// return degree {-180 ~ +180}
	slope = atan2((y2 - y1),(x2 - x1)) * 180 / PI;  
	slope = (slope >= 0)? slope : 180 + slope; // 0~180 degree
	return slope; 
}

int Line::cal_position(){  // ***********
	/*
	front, right = return 2
	back, left = return 1
	*/
	if((slope > 90-3 && slope < 90+3 )){ // vertical
		return (x1>600)? 2:1; // 1: left,  2: right
	}
	else if(slope > (0-3)+180 || slope < 0+3){  // horizontal
		int y_min = min(y1,y2);
		return (y1>600)? 2:1; // 1: back,  2: front
	}
	else return 0; // not (vertical or horizontal)
	
} 

// 2. Class Calculate_lines ----------------------------------------------------------------------------------------------------------------------------
class Cal_lines{
	public:
		int num_lines;
		Line line[18];
		int num_1, num_2, num_11, num_12, num_21, num_22; 

		Cal_lines();
		void append(Line line_t);
		void labeling_wall();
		void labeling_obstacle();
		void howmany_num();

		// Driving Algorithm
		int what_case(); 
		void drive_case1(int *velocity, int *steering, int case_is);
		void drive_case2(int *velocity, int *steering, int case_is);
		void drive_case3(int *velocity, int *steering, int case_is);
};

Cal_lines::Cal_lines(){
	num_lines = 0;
	num_1 = num_2 = num_11 = num_12 = num_21 = num_22 = 0;
}

void Cal_lines::append(Line line_t){  
	// Have to change IF condition 85~95...
	if( (line_t.slope > 90-3 && line_t.slope < 90+3 ) || (line_t.slope > (0-3)+180 || line_t.slope < 0+3) ){
		if(num_lines<18){
			line[num_lines] = line_t;
			num_lines++;
		}
	}
}

void Cal_lines::labeling_wall(){ // ***********
	for (int i = 0; i < num_lines; i++)
	{
		line[i].label= 0; //??

	}
}

void Cal_lines::labeling_obstacle(){ // ***********
	for (int i = 0; i < num_lines; i++)
	{
		line[i].label= 0; //??

	}
}

void Cal_lines::howmany_num(){ 
	for (int i = 0; i < num_lines; i++)
	{
		if(line[i].label == 1)       num_1++;
		else if(line[i].label == 2)  num_2++;
		else if(line[i].label == 11) num_11++;
		else if(line[i].label == 12) num_12++;
		else if(line[i].label == 21) num_21++;
		else if(line[i].label == 22) num_22++;
		else cout << "line[i]'s label has to be 1,2,11,12,21,22";

	}
}

int Cal_lines::what_case(){ // ***********
	return 1;
}

void Cal_lines::drive_case1(int *velocity, int *steering, int case_is){ // ***********

}

void Cal_lines::drive_case2(int *velocity, int *steering, int case_is){ // ***********

}

void Cal_lines::drive_case3(int *velocity, int *steering, int case_is){ // ***********

}



// 3. Class detected_lines ------------------------------------------------------------------------------------------------------------------------
class Detected_lines{
	public:
		Line line_1[3];
		Line line_2[3];
		Line line_11[3];
		Line line_12[3];
		Line line_21[3];
		Line line_22[3];
		int counted_1, counted_2;
		int counted_11, counted_12;
		int counted_21, counted_22;
		void append(Line line_t);
		Detected_lines();

};

Detected_lines::Detected_lines(){
	counted_1 = counted_2 = counted_11 = counted_12 = counted_21 = counted_22 = 0;
}

void Detected_lines::append(Line line_t){
	switch (line_t.label){
		case 1:
			line_1[counted_1] = line_t;
			counted_1++;
			if (counted_1 == 3) counted_1 = 0;
			break;
		case 2:
			line_2[counted_2] = line_t;
			counted_2++;
			if (counted_2 == 3) counted_2 = 0;
			break;
		case 11:
			line_11[counted_11] = line_t;
			counted_11++;
			if (counted_11 == 3) counted_11 = 0;
			break;
		case 12:
			line_12[counted_12] = line_t;
			counted_12++;
			if (counted_12 == 3) counted_12 = 0;
			break;
		case 21:
			line_21[counted_21 %3] = line_t;
			counted_21++;
			if (counted_21 == std::numeric_limits<int>::max()) counted_21 = 300;
			break;
		case 22:
			line_22[counted_22 %3] = line_t;
			counted_22++;
			if (counted_22 == std::numeric_limits<int>::max()) counted_22 = 300;
			break;
		default:
			cout << "line's label must be one of 1,2,11,12,21,22" << endl;
	}
}

Detected_lines global_lines;

// Main -------------------------------------------------------------------------------------------------------------------------------------------------------------
int main()
{
	// 컬러 이미지를 저장할 Mat 개체를 생성합니다.

	cv::Mat img(1200, 1200, CV_8UC3, cv::Scalar(0,0,0));
	cv::Mat img_fliped(1200, 1200, CV_8UC3, cv::Scalar(0,0,0));
	int arr[5][4] = 
					{ {450,800,450,450}, //0
				    	{750,800,750,450}, //1
				    	{630,750,680,750}, //2
				    	{610,780,630,750}, //3
				    	{1100,900, 450,900} //4
					};
	
	line(img, Point(arr[0][0],arr[0][1]), Point(arr[0][2],arr[0][3]), Scalar(255,255,255),1); // 0
	line(img, Point(arr[1][0],arr[1][1]), Point(arr[1][2],arr[1][3]), Scalar(255,255,255),1); // 1
	line(img, Point(arr[2][0],arr[2][1]), Point(arr[2][2],arr[2][3]), Scalar(255,255,255),1); // 2
	line(img, Point(arr[3][0],arr[3][1]), Point(arr[3][2],arr[3][3]), Scalar(255,255,255),1); // 3
	line(img, Point(arr[4][0],arr[4][1]), Point(arr[4][2],arr[4][3]), Scalar(255,255,255),1); // 4
	circle(img, Point(600,600),2,Scalar(255,0,0),2);
	

	Cal_lines lines;
	for(int i = 0; i < 5; i++ ){
		Line temp_line(arr[i][0],arr[i][1],arr[i][2],arr[i][3]);
		lines.append(temp_line);
	}

	lines.labeling_wall();
	lines.labeling_obstacle();

	int velocity = 1111;
	int steering = 2222;
	int case_is = lines.what_case();
	if(case_is == 1)      lines.drive_case1(&velocity, &steering, case_is);
	else if(case_is == 2) lines.drive_case2(&velocity, &steering, case_is);
	else if(case_is == 3) lines.drive_case3(&velocity, &steering, case_is);
	else if(case_is == 4) lines.drive_case1(&velocity, &steering, case_is);
	else if(case_is == 5) lines.drive_case2(&velocity, &steering, case_is);
	else if(case_is == 6) lines.drive_case3(&velocity, &steering, case_is);
	else cout << "case_is is not valid";

	
	float data = float(velocity) + float(steering+1) / 10000;
	cout << fixed;
	cout.precision(5);
	cout << "Final ToArduData.data is :" << data << endl;

	cout << lines.num_lines << endl;
	cout << lines.line[0].position << endl;

	flip(img,img_fliped,0);
	imshow("result", img_fliped);
	waitKey(0);
	//waitKey(1);
}
```
