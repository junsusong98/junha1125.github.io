---
layout: post
title: 【Detection】Feature Pyramid Transformer
---

- **논문** : [Feature Pyramid Transformer](https://arxiv.org/abs/2007.09451)   
  논문 필기는 `C:\Users\sb020\OneDrive\21.1학기\논문읽기_21.1` 여기 있으니 참조
- **분류** : Detection
- **읽는 배경** : 
- **느낀점** : 
  1. 이 논문도 약간 M2Det같다. 뭔가 오지게 많이 집어넣으면 성능향상이 당연하긴하지.. 약파는 것 같다. 
  2. 비교도 무슨 Faster-RCNN, Mask-RCNN 이런거 써서 비교하는게 전부이고 약간 많이 부족하고 아쉬운 논문이다.



---

---

# Feature Pyramid Transformer

![image-20210325162755839](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210325162755839.png?raw=tru)

- 이 그림이, 이 논문의 전부. FPN의 결과에 **Same size + richer contects** 를 가지는 feature map 만드는 것을 목표로 한다.
- 보라색 동그라미 부분과 같이, (1) feature map 그대로 self-attention을 수행하는 Self-transformer (2) Up! 하는 Rendering Transformer (3) Down! 하는 Grounding transformer 를 제안했다. (개인적인 생각으로, 이 방법이 약간 어설프고? 약간 너무 파라메터와 레이어를 무작정 늘리는 행동? 같다.) 



# 1. Conclusion, Abstract

- efficient feature interaction approach
- 3개의 Step : `Encoder(Self-Transformer)`, `Top-down(Grounding Transformer)`, `Bottom-up(Rendering Transformer)`
- FPN(`feature pyramid network`)에서 나온 P2~P5에 FPT(`feature pyramid transformer`)를 적용해서 P2~P5는 **크기는 보존**되는데 좀더 **Sementic한** Feature map이 되게 만든는 것을 목표로 한다.(`the same size but with richer contexts`) 따라서 이 모듈은 easy to plug-and-play 한 모델이다. (대신 파라메터수가 엄청 많아 진다. 결과 참조)
-  the non-local spatial interactions (2017년에 나온 논문으로 MHSA과 비슷한 구조를 가지고 있지만 좀 다르다 예를들어서 Multi head를 안쓴다던지...)는 성능향상에는 도움이 된다. 하지만 across scale 정보를 이용하지 않는다는 문제점이 있다. 
- 이 문제점을 해결하고자 interaction across both space and scales을 수행하는 `Feature Pyramid Transformer (FPT)` 모듈을 만들었다.



---

# 2. Instruction, Relative work

![image-20210325162730412](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210325162730412.png?raw=tru)

- 위의 그림이 개같으니 굳이 이해하지 못해도 상관하지 말기
- Fig. 1 (d) :  non-local convolution을 통해서 상호 동시 발생 패턴 (reciprocal co-occurring patterns of multiple objects)을 학습할 수 있다고 한다. 예를 들어서, 컴퓨터가 이미지에 있으면 주변에 책상이 있는게 옮바르지, 갑자기 도로가 예측되는 것을 막는다고 한다. (Self-attention 구조를 약간 멋지게 말하면 이렇게 표현할 수 있는건가? 아니면 ` non-local convolution`논문에 이러한 표현을 하고 증명을 해놓은 건가? 그거까지는 모르겠지만 ` non-local convolution`은 Transformer 구조에 진것은 분명한 것 같다.)
- Fig. 1 (e) : 이와 같이 cross scale interactions을 유도할 것이라고 한다. 
- *Feature Pyramid Transformer (FPT) enables features to interact across space and scales. 내가 지금 부정적으로 생각해서 그렇지, 파라메터 오지게 많이 하고 깊이 오지게 많이 한다고 무조건 성능이 올라가는 것은 아니다. 그렇게 해서라도 성능이 올랐으니 일단은 긍정적으로 봐도 좋다. 하지만 FPN을 차용해서 아이디어를 짠것은 좋은데 좀더 깔끔하고 신박하고 신기하게 설계한 구조와 방법이 필요한 듯 하다.*



---

# 3. Method

![image-20210325162755839](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210325162755839.png?raw=tru)

---

## 3.1 Non-Local Interaction Revisited

- `typical non-local interaction` 은 다음과 수식으로 이뤄진다고 할 수 있다. (하나의 level Feature map에 대해서)  하지만, 논문에서는 이 공식을 그대로 사용하지 않는다.     
  ![image-20210325164245529](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210325164245529.png?raw=tru)
- (사실 self-attention과 거의 같은 구조이다. 차이점에 대해서는 BottleneckTransformer를 참조하면 도움이 될 수 있다.)



---

## 3.2 Self-Transformer

- 위의 `typical non-local interaction`공식에서 하나만 바꿔서, 새로운 이름을 명명했다. 
- 위에서 weight를 계산하는것이 그냥 softmax를 사용했다. 이것을 `the Mixture of Softmaxes (MoS) [34]` 로 바꾼다.    
  ![image-20210325165146471](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210325165146471.png?raw=tru)
- 위의 N에 대한 설명은 자세히 나와있지 않다. `the same number of divided parts N` 이라고 나와있는게 전부이다. 따라서 위에 내가 생각한게 맞는지 잘 모르겠다.  [34] 논문을 참고하면 알 수도 있다.



---

## 3.3  Grounding Transformer

![image-20210325165710081](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210325165710081.png?raw=tru)

- a top-down non-local interactio을 수행하는 방법이 위와 같다. 
- 어렵게 생각할거 없고 위에 그림과 같이 q, k, v를 설정하여 self-attention을 적용한 모듈이다. 
- 예를 들어서 변수 = 변수.shape로 표현해 정리한다면, `q = H * W * d`,  `K = h * w * d`  차원을 가진다고 하면, `q_i = 1 * d`, `k_j = 1 * d` 가 된다고 할 수 있다. d를 맞추는 것은 channel 크기를 맞춰주면 되는 것이기 때문에 그리 어려운 문제는 아니다.
- **Locality-constrained Grounding Transformer** : 그냥 Grounding Transformer를 적용하는 방법도 있고, Locality 적용하는 방법이 지들이 제안했다고 한다. stand-alone에 있는 내용아닌가...



---

## 3.4 Rendering Transformer

- a bottom-up fashion. self attentino을 적용한 방법이 아니다. 과정은 아래와 같다. (논문의 내용을 정리해놓은 것이고 헷갈리면 논문 다시 참조)     
  ![image-20210325170353850](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210325170353850.png?raw=tru)



---

## 3.5 Overall Architecture

- FPT for object detection and instance segmentation
  - BFP = FPN 지들맘대로 이름 바꿔서 사용함.
  - `divided parts of N`은 ST에서는 2 그리고 GT에서는 4로 설정했다.
  - FPT를 통해서 나오는 Pyramid Feature map들을 head networks에 연결되어서 예측에 사용된다.
  - head networks는 Faster R-CNN 그리고 Mask RCNN에서 사용되는 head를 사용한다.
  - (분명 retinaNet과 같은 head도 해봣을텐데, 안 넣은거 보니 성능이 그리 안 좋았나? 라는 생각이 든다.)
- FPT for semantic segmentation.
  - dilated ResNet-101 [4]
  - Unscathed Feature Pyramid (UFP)  - a pyramidal global convolutional network [29] with the internal kernel size of 1, 7, 15 and 31
  - segmentation head network, as in [14 `ASPP`,19]



---

# 5. Results

![image-20210325171045436](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210325171045436.png?raw=tru)



![image-20210325171111809](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210325171111809.png?raw=tru)




