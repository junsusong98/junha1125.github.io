---
layout: post
title: 【Detection】Understanding YOLOv4 paper with code w/ my advice
---

- **논문** : [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)
- **분류** : Object Detection
- **저자** : Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao
- **읽는 배경** : Recognition Basic. Understand confusing and ambiguous things.
- **읽으면서 생각할 포인트** : 코드와 함께 최대한 완벽히 이해하기. 이해한 것 정확히 기록해두기.
- **느낀점**  : 
  - dd
- 목차
  1. YoloV4 from youtube ([바로가기](https://junha1125.github.io/blog/artificial-intelligence/2021-02-08-YoloV4withCode/#1-yolov4-from-youtube))
  2. YoloV4 Paper ([바로가기](https://junha1125.github.io/blog/artificial-intelligence/2021-02-08-YoloV4withCode/#2-yolov4-paper))
  3. Code - Tianxiaomo/pytorch-YOLOv4 ([바로가기](https://junha1125.github.io/blog/artificial-intelligence/2021-02-08-YoloV4withCode/#3-tianxiaomopytorch-yolov4))





# 1. YoloV4 from youtube

- [youtube 논문 발표 링크](https://www.youtube.com/watch?v=CXRlpsFpVUE) - 설명과 정리를 잘해주셨다.
- 이 정도 논문이면, 내가 직접 읽어보는게 좋을 것 같아서 발표자료 보고 블로그는 안 찾아보기로 함
- 강의 필기 PDF는 “OneDrive\21.겨울방학\RCV_lab\논문읽기”

![img01](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-01.png?raw=true)
![img02](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-02.png?raw=true)
![img03](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-03.png?raw=true)
![img04](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-04.png?raw=true)
![img05](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-05.png?raw=true)
![img06](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-06.png?raw=true)
![img07](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-07.png?raw=true)
![img08](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-08.png?raw=true)
![img09](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-09.png?raw=true)
![img10](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-10.png?raw=true)
![img11](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-11.png?raw=true)
![img12](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-12.png?raw=true)
![img13](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-13.png?raw=true)
![img14](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-14.png?raw=true)
![img15](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-15.png?raw=true)
![img16](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-16.png?raw=true)
![img17](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-17.png?raw=true)
![img18](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-18.png?raw=true)
![img19](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-19.png?raw=true)
![img20](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-20.png?raw=true)
![img21](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-21.png?raw=true)
![img22](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-22.png?raw=true)
![img23](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-23.png?raw=true)
![img24](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-24.png?raw=true)
![img25](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-25.png?raw=true)
![img26](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-26.png?raw=true)
![img27](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/yolov4_youtube/yolov4_youtube-27.png?raw=true)





# 2. YoloV4 Paper





# 3. Tianxiaomo/pytorch-YOLOv4

1. Github Link : Tianxiaomo/[pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)

   







