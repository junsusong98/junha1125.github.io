---
layout: post
title: 【Domain】Adversarial Discriminative Domain Adaptation = ADDA 
---

- **논문** : [Large-Scale Long-Tailed Recognition in an Open World - y2019-c103](https://arxiv.org/pdf/1904.05160.pdf)

- **분류** : Unsupervised Domain Adaptation

- **저자** : Ziwei Liu1,2∗ Zhongqi Miao2∗ Xiaohang Zhan1

- **읽는 배경** : (citation step1) Open Componunt Domain Adaptation에서 Memory 개념이 이해가 안되서 읽는 논문. 

- **읽으면서 생각할 포인트** : 논문이 어떤 흐름으로 쓰여졌는지 파악하자. 내가 나중에 쓸 수 있도록.

- **[동영상 자료](https://www.youtube.com/watch?v=A45wrs1g8VA)** 
  - <img src="C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210126160627433.png" alt="image-20210126160627433" style="zoom:80%;" />

- **느낀점**  
  1. Instruction이 개같은 논문은..
     - abstract 빠르게 읽고, Introduction 대충 읽어 넘겨야 겠다. 뭔소리하는지 도저히!!!!!! 모르겠다. 
     - 지내들이 한 과정들을 요약을 해놨는데.. 나는 정확히 알지도 못하는데 요약본을 읽으려니까 더 모르겠다.
     - 따라서 그냥 abstract읽고 introduction 대충 모르는거 걍 넘어가서 읽고. 
     - relative work의 새로운 개념만 빠르게 훑고, 바로 Ours Model에 대한 내용들을 먼저 깊게 읽자. 그림과 함께 이해하려고 노력하면서. 
     - 그리고! Introduce을 다시 찾아가(👋👋) 읽으며, 내가 공부했던 내용들의 요약본을 읽자 
  2. 아무리 Abstract, Instruction, Relative work를 읽어도, 이해가 되는 양도 정말 조금이고 머리에 남는 양도 얼마 되지 않는다. 지금도 위 2개에서 핵심이 뭐였냐고 물으면, 대답 못하겠다. 
     - 현재의 머신러닝 논문들이 다 그런것 같다. 그냥 대충 신경망에 때려 넣으니까 잘된다. 
     - 하지만 그 이유는 직관적일 뿐이다. 따라서 대충 이렇다저렇다 삐까뻔쩍한 말만 엄청 넣어둔다. 이러니 이해가 안되는게 너무나 당연하다. 
     - 이런 점을 고려해서, 좌절하지 않고 논문을 읽는 것도 매우 중요한 것 같다. (👋👋)여기 아직 안읽었다고??? 걱정하지 마라. 핵심 Model 설명 부분 읽고 오면 더 이해 잘되고 머리에 남는게 많을거다. 화이팅.

- **다 읽은 후, 필수로 읽어야 겠다고 생각이 든 논문**




## 0. Abstract

- the present & challenges
  1. Real world data often have a long-tailed and open-ended distribution. 즉 아래의 그래프의 x축은 class종류를 데이터가 많은 순으로 정렬한 것이고, y축은 해당 클래스를 가지는 데이터 수. 라고 할 수 있다. Open Class는 우리가 굳이 Annotate 하지 않는 클래스이다.
  2. <img src="C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210126154310688.png" alt="image-20210126154310688" style="zoom: 80%;" />
- Ours - 아래 내용 요약



## 1. Introduction

-  the past	
   -   대부분의 기법들은 Head Class 인식에 집중하지만,   
      최근 Tail class 문제를 해결하기 위한 방법이 존재한다. (=Few-shot Learning)for the small data of tail classes [52, 18]
   
- Ours
   -  우리의 과제(with one integrated algorithm)  
      1) imbalanced classification    
      2) few-shot learning  
      3) open-set recognition  
      즉. tail recognition robustness and open-set sensitivity:
      
   - Open Long-Tailed Recognition(OLTR) 이 해결해야하는 문제  
     1) how to **share visual knowledge(=concept)** between **head and tail** classes (For **robustness**)  
     2) how to **reduce confusion** between **tail and open** classes (For **sensitivity**)     

   - 해결방법 

     - <u>2번째 페이지 We develop an OLTR 문단부터 너무 이해가 어렵다. 따라서 일단 패스하고 다시 오자. 요약본이니, 구체적으로 공부하고 나면 이해하기 쉬울 거다. </u> 👋👋👋

     - ```sh
       # 이전에 정리해 놓은 자료. 나중에 와서 다시 참조.
       1 mapping(=dynamic meta-embedding) an image to a feature space  = **a direct feature** computed(embeded) from the input image from the training data   
       2) visual concepts(=visual memory feature) 이 서로서로 연관된다.(associate with) = A visual memory holds **discriminative centroids**     
       3) A summary of **memory activations** from the direct feature   
       그리고 combine into a meta-embedding that is enriched particularly for the tail class.
       ```

   

   ## 2. Related Works

   ![image-20210126181200150](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210126181200150.png)

- Imbalanced Classification
  - Classical methods -  under-sampling head classes, over-sampling tail classes, and data instance re-weighting.
  -  Some recent methods - metric learning, hard negative mining, and meta learning.
  - Ours
    - combines the strengths of both metric learning[24, 37] and meta learning[17, 59]
    - Our dynamic meta-embedding👋👋👋
- Few-Shot Learning
  -  초기의 방법들 they often suffer a moderate performance drop for head classes.
  - 새로운 시도 : The few-shot learning without forgetting,  incremental few-shot learning.
  - 하지만 : 위의 모든 방법들은, the training set are balanced.
  - In comparison, urs : 👋👋
- Open-Set(new data set) Recognition
  -  OpenMax [3] : calibrating the output logits
  - OLTR approach incorporates👋👋



## 3. Our OLTR Model

![image-20210126184016363](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210126184016363.png)

- 우리 모델의 핵심은 Modulated Attention 그리고 Dynamic meta-embedding 이다. 
  - **Embedding** 부분은 visual concepts between **head and tail** 과 관련깊고
  - **Attention** discriminates(구분한다) between **head and tail**
  - **reachability** separates **tail and open** classes
- Our OLTR Model
  - 