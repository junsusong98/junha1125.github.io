---
la 
yout: post
title: 【Detection】Understanding Cascade R-CNN paper with code 
---

- **논문** : [Cascade R-CNN: Delving into High Quality Object Detection](https://openaccess.thecvf.com/content_cvpr_2018/papers/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.pdf)

- **분류** : Object Detection

- **저자** : Zhaowei Cai, Nuno Vasconcelos

- **읽는 배경** : Recognition Basic. Understand confusing and ambiguous things.

- **읽으면서 생각할 포인트** : 코드와 함께 최대한 완벽히 이해하기. 이해한 것 정확히 기록해두기.

- **느낀점**  : 

  - 최근 논문일수록, 블로그나 동영상의 내용 정리가 두리뭉실하고 이해하기 힘들다. 디테일성은 거의 없다. 따라서 그냥 논문 읽는게 최고다. 만약 블로그에 내용 정리가 잘 되어 있다고 하더라도, 내가 블로그 내용에 대한 신뢰가 안가서, 합리적 추론의 이해(덜 디테일한 설명은 나의 생각을 좀 더 추가해서 이해하는 과정)을 할 수도 없다. 따라서 **논문이나 읽자. 시간 낭비하지 말고.**
  
  



# 1. Cascade R-CNN

## 1-(1). 참고 자료 

1. (1) [Cascade R-CNN 블로그](https://blog.lunit.io/2018/08/13/cascade-r-cnn-delving-into-high-quality-object-detection/), (2) [Cascade R-CNN Youtub동영상1](https://www.youtube.com/watch?v=1_-HfZcERJk&feature=youtu.be)

2. 참고 자료 소감 : 

   - 블로그 : 너무 result 해석에 많은 초점을 둔다. 물론 단순한 방법이라지만.. 그리 아름답게 정리해놓은 글인지는 모르겠다. 내용도 약간 두리뭉술해서 나에게는 이해가 어렵다. 논문이나 읽자.
   - 동영상 : 핵심이 Casecade-Mask RCNN이다. 디테일이 거의 없다. 

3. 참고자료 내용 정리 : 

   1. 이 논몬은 object detector의 약점을 잘 파악했다. 이 논문을 통해, 아주 복잡한 방법을 적용해서 성능 향상을 미약하게 이루는 것 보다는, **문제점을 잘 파악하기만 하면 어렵지 않게 성능을 향상시킬 수 있음을 보여준다.**

   2. 문제점 파악하기

      1. Introduction에 실험 그래프가 나온다. 블로그 글은 이해 안되서, 논문 읽는게 낫겠다.    

         <img src="C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210204160324773.png" alt="image-20210204160324773" style="zoom: 80%;" />

   3.  Cascade R-CNN은 위 사진의 오른쪽 그림의 구조이다. 굳이 설명하지 않겠다. 특히 단계를 거듭할 수록 **보다 더 높은 IOU를 기준으로 학습시킨다.**

      - 과거 방식 (1) 단순하게 하나의(같은) classifier를 Iterative하게 사용하는 것은 큰 성능 향상을 주지 못한다.  (2) 서로 다른 classifier를 여러개 만들고, 각각의 IOU기준을 다르게 주고 학습을 시키는 방법도 성능 향상은 그저 크지 않다. 즉 여러 classifier의 ensenble 방법이다.
      - Cascade R-CNN은 각각의 classifier는 각각 Threshold 0.5, 0.6, 0.7 로 학습. 예측 bounding box와 GT-box가 겹치는 정도가 Threshold 이상이여야지만, 옳게 예측한 것이라고 인정해줌. 이하라면 regressing 틀린거로 간주하고 loss를 준다. 



## 1-(2). Paper Review 

1. Conclustion	
   - dd





# 2. guoruoqian/cascade-rcnn_Pytorch

1. Github Link : [guoruoqian/cascade-rcnn_Pytorch](guoruoqian/cascade-rcnn_Pytorch)









