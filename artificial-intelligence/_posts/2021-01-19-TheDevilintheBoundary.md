---
layout: post
title: 【segmentation】The Devil Boundary for Instance Segmentation w/ advice
---

- **논문** : [The Devil is in the Boundary: Exploiting Boundary Representation for Basis-based Instance Segmentation](https://arxiv.org/abs/2011.13241)

- **동영상** : [https://www.youtube.com/watch?v=XvLo5WrtHu0](https://www.youtube.com/watch?v=XvLo5WrtHu0) -> why, how, what in the past 에 대한 내용들이 잘 담겨 있으니 꼭 참고.

- **저자** : Myungchul Kim, Sanghyun Woo, Dahun Kim, In So Kweon 

- **읽는 배경** : 현재 Domain adapation, Self-supervise, Transformer 등에 관심이 가지만, 그래도 가장 구체적이며 많은 지식을 가지고 있는, Segmentation이나 Object Detection과 관련된 이 논문을 먼저 읽어보고 싶었다. 같은 랩 석사 선배님이 적으신 논문으로써, 내가 1년 안에 이런 논문을 만들어야 한다는 마음 가짐으로 논문을 읽어보려고 한다. 

- **읽으면서 생각할 포인트** : Reference를 어떻게 추가했나? 실험은 어떤 것을 어떻게 하였나? Relative work는 어느정도로 작성하였나? 과거 지식은 어느 정도가 필요한가? 코드 개발은 어떻게 하였나? 

- **느낀점**  

  1. 논문 안에는 핵심 내용(global 사용, score 사용) 등이 있는데, 최근 논문들의 핵심 내용만 쏙쏙 캐치해서 그것들을 잘~융합해서 개발이 되었다. -> 이런 논문 작성 방식도 추구하자. 
  2. 논문 많이 읽어야 겠다... 완벽하게 이해는 안되고, 60% 정도만 이해가 간다. 필수적으로 읽어야 하는 논문 몇가지만 읽고 나면 다 이해할 수 도 있을 듯 하다. 지금은 전체다 이해가 안된다고 해도 좌절하지 말자.

  3. 정말 많은 노력이 보였다. 내가 과연 이정도를 할 수 있을까? 라는 생각이 들었지만 딱 한달 동안 이와 관련된 매일 논문 1개씩 읽는다면, 잘하면 좀더 창의력과 실험을 가미해서 더 높은 성능을 내는 모델을 만들 수 있지 않을까? 하는 생각은 든다. 따라서 하나의 관심 분야를 가지고 그 분야의 논문 20개는 읽어야 그 쪽은 조금 안다고 할 수 있을 것 같다.

  4. 만약 Segmentation을 계속 하고 싶다면, 아래의 '필수 논문'을 차례대로 읽도록 하자. 

  
 

---

---

## 질문&답변 ⭐⭐​

- 질문
  1.	논문 내용에 대한 질문은 없다. 왜냐면 내가 찾아 읽는게 먼저이기 때문이다. 필수로 읽어야 겠다고 생각한 논문들을 먼저 읽고 모르는 걸 질문해야겠다. 
  2.	현재도 Segmentation을 매인 주제로 연구하고 계신지? 아니면 다른 과제나 연구로 바꾸셨는지?논문을 준비하신 기간? 
  3.	이 분야를 항상 관심을 두고 계셨던 건지? 그래서 그 분야의 논문을 항상 읽어오신 건지?
  4.	논문을 준비하신 기간? 
  5.	코드 제작은 어떻게 하셨는지? 어떤 코드 참고 및 어떤 방식으로 수정? 몇 퍼센트 수정 및 추가?
  6.	아무래도 saturation된 Instance Segmentation에 대해서 계속 공부해나가시는거에 대한 두려움은 없으신지?
  7.	석사기간에 이런 좋은 논문을 쓰는게 쉽지 않은데... 혹시 팁이 있으신지?
- 선배님답변 간략히 정리 (정말 감사합니다)
  1. 공부에는 **왕도가 없다**. 어떤 문제와 관심분야를 하나로 정해두고 끝장을 보겠다고 해도 끝이 없다.
  2. 일정 분야에 대해서 **흥미**를 가지는 마음가짐이 굉장히 중요하다. 이 마음이 사라지고 의무와 책임만 남는 상황이라면 진정한 나만의 연구, 나만의 결과, 행복한 연구, 행복한 생활을 이뤄나갈 수 없다. 
  3. 조급함 보다는 **꾸준함**이 중요하다. 꾸준하게 조금씩, **라이프 밸런스**를 맞춰가며 취미도 해가며 **흥미롭게 나의 연구와 공부**를 해 나가는게 중요하다. 연구와 공부에 대한 의무감이 생기면 좋은 연구, 좋은 결과를 낼 수 없다. 
  4. **왜 이런 논문이 나왔지? 지금까지의 계보는 어떻고, 그런 계보가 나온 이유가 무엇이지? 어떤 개념과 어떤 문제, 어떤 정의에 의심을 가지고 시작된 연구이지?** 라는 생각을 가지고 논문을 읽고 공부하는 것이 굉장이 중요하다. 예를 들어서 mask score이 나온 이유는 뭐지? boundary score가 사실은 더 중요한거 아닐까? mask-rcnn에서 masking이 왜 잘되지? BB가 이미 잘 만들어지기 때문에? 이걸 없애야 하지 않을까?? 그렇다면 어떻게 없애야지? 이 loss, 이 score가 과연 맞는 건가? 이런 **의심, 질문, 반박**을 찾아 논문을 읽고, 나도 이런 마음을 가지고 생각하는 것이 굉장히 중요하다.
  5. 마음을 잡고 논문을 준비한 기간은, 6개월 7개월. 코드 작성도 걱정 할거 없다.
  6. 다양한 걱정, 고민, 근심이 드는 것은 너무 당연한 생각이다. **너무 걱정하지말고 꾸준히 해나간다면 분명 잘 할 수 있다.**
  7. recognition, reconstruction, reorganization 이라는 3가지 3R 문제가 딥러닝에서 있다. 특히 recognition을 잘 따라간다면, 어떤 문제에서도 적용되어 사용하는 것을 알 수 있다. 
  8. classification 부터 object detection, segmentation 에 대한 계보들을 보면서, **왜??** 라는 생각을 가지고 하나하나를 바라보아야 한다. 이건 왜 쓰였지? 이건 왜 나온거지? 이건 왜 좋지? 이러다보면 정말 과거 계보부터 봐야하기도 하지만 그것도 나의 선택이다. 정말 필요하다고 생각이 들면 찾아서 공부하면 되는거고, 아니다 다른게 더 중요하다 하면, 다른거 새로운 기술을 공부해도 좋은거다. 정답은 없다. 그건 **나의 선택이고 걱정하지 말고 흥미를 가지고 꾸준하게 하면 되는 거다.** 
- 다 읽은 후, 필수로 읽어야 겠다고 생각이 든 논문
  - YOLACT [4]-2019
  - BlendMask [6] - 2020
  - Boundary-preserving Mask RCNN[11] - 2020
  - Mask scoring r-cnn - 2019
  - Zimmermann et al.[47] - auxiliary edge loss. Boundary Learning - 2019 
  - Single shot instance segmentation with point representation [global-area-based methods] 2020
  - a Dice metric [13] 
  - CondInst  [[37](https://arxiv.org/pdf/2003.05664.pdf)]
  - basis 개념 [4, 5, 38, 6, 37] 중에 적당한거 하나



---

---

# The Devil is in the Boundary

## 1. Abstract, Introduction

- the present and problems
  - most instance segmentation 의 문제점 
    1. two-stages-instance-segmentation은 first-stage's bounding box에 mask결과값이 많이 의존된다(step-wise).  
    2. 성능이 구리다. region-specific information 만을 사용하고, ROI-pooled features 를 사용하기 때문이다. 이 두가지 문제를 아래의 것들로 해결했다.
  - 최근까지 자주 사용되고 있는 [4, 5, 38, 6, 37] 
    1. basis framework : global image-level information 을 사용한다. 
    2. 하지만 정확한 지도학습적 방법으로, 적절한 Loss를 사용하서, global mask representations 를 학습하진 않는다. (?) last mask output signal에 의해서 학습 되므로.
  - boundary 관점에서 집중하는 최근 논문으로는 Boundary-preserving Mask RCNN (2-stage) 논문도 있다. a boundary prediction head in 'mask head' 를 사용한다. 
- <u>우리의 방법</u>
  1. **a boundary prediction head** 를 추가적으로 사용했다.
  2. **holistic image-level** instance boundaries( = global boundary representations ) 을 학습한다. 그러기 위해서 우리는 **the boundary ground-truths** 를 사용한다. 
  3. 새로운 측정지표로써 **novel instance-wise boundary-aware mask score** 를 사용했다. 
  4. **Blend-Mask**(strongest single-stage instance segmentation methods)를 보완하여 만듬
  5. **state-of-the art** methods on the COCO dataset.



---

## 2. Related Work

- two-stage

  1. Mask RCNN : RoIAlign 
  2. PANet : FPN+Mask 
  3. Mask scoring RCNN (읽어야함): alignment between the confidence score and localization accuracy of predicted masks

  - 문제점 :  (1) RoI pooling에 의한 정보 손실 (2) quite slow

- Single-stage 

  - pixel-wise predictions of cues 픽셀 관점 예측 단서들 : (a) directional vectors (b) pairwise affinity (c) watershed energy (d) embedding learning (=  local-area-based methods, 각각에 대한 설명은 각 paper 참고) -> and then grouping pixel-wise predictions for object instances  -> not good ....
  - **global-area-based** methods 탄생 [4, 6, 38] : 
    1. Generate intermediate FCN feature maps, called ‘**basis**’. 
    2. Assemble the extracted basis features.
  
- **Boundary learning** for instance segmentation

  - the past and problems
  - CASENet : sementic segmentation.  category-aware boundary detection.
    - InstanceCut : instance-level.
    - 이 둘은 expensive! (super-pixel extractions and grouping computations 때문에)
  - the present and problems
    - Zimmermann et al.[47]:  Boundary Learning,  auxiliary edge loss.
    - Boundary-preserving mask r-cnn[11] : instance-level boundaries head ([47]에서 업그레이드)
    - 하지만 위의 2가지 방법은 two-stage methods(RoI-wise)! 이다. ROI는 local!적이므로 global 정보를 이용하지 못한다. 



---

## 3. Background (single-stage)

- (global-area-based) Basis-based instance segmentation의 시작 :     
  ![image](https://user-images.githubusercontent.com/46951365/110077422-6e751f00-7dc9-11eb-9eff-6af4f12e0555.png)
  1. Basic Head(=Protonet) : FCN으로 a set of basis mask representations 를 생성한다. 
  2. Detection Head(=Prediction head) : Detected Box를 찾는다. 즉 **instance-level parameters(= instance-specific information)**를 예측한다.
  3. (1과 2를 결합해서) instance segmentation을 수행한다. 
- 위의 pipeline을 따르는 모델들
  1. **YOLACT** :  32 global base, the according instance-specific scalar coefficients, a linear combination among the bases.  --->  ~~cause [rich instance features **vs** effective assembly methods\]~~
  2. improved the assembly parameters : **BlendMask** [6], SipMask [5], and **CenterMask** [38] 
  3. instance-specific representation : **CondInst** [37] (Key word: dynamic filter weights, sliding window, the predicted convolution weight filters)



---

## 4. Boundary Representation

- **B2Inst**
  1. **backbone** for feature extraction
  2. instance-specific detection head (**Detection head**)
  3. the global basis head,  the global image information (**Basis head**)
  4. Bundary Basis = **mask scoring head** 
  5. (2번)과 (3번)을 **결합**해서, final masks 예측.
  6. ~~BlendMask instantiation (boundary basis)~~

<img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210119165307132.png?raw=tru" alt="image-20210119165307132" style="zoom:80%;" />

- Details
  1. **basis head** 
     - FPN feature를 받고, a set of K(=#Bases) global basis features를 생성한다 
     - 4개의 3x3 conv 통과 + upsampling  + reduce the channel-size
     - The previous basis head is supervised by the last mask loss only. (?)
  2. **Detection Head**
     - Basic head와 병렬로, instance-level parameters 들을 추측한다. 
     - 특히 여기서 나오는 attention map feature는  **Boundary Basic** (BlendMask 참고)에서 사용된다.
  3. Loss 함수
     - 1) binary cross-entropy loss   
     - 2) dice loss 
     - 3) boundary loss 
  4. image boundary (그림에 없음)
     - **a holistic boundary of all instances** (= global boundary representations) in a scene (instance 하나하나 아니라)
     - Overlapping objects and complex shapes 문제에서 좋은 성능을 가져다 준다.
     - the boundary supervision 은 어렵지 않다. mask annotations를 그냥 이용하면 되므로. the binary mask ground-truths에서 soft boundaries를 찾기 위해서, Laplacian operator 를 사용한다.
- Boundary-aware mask scoring
  1. Mask scoring R-CNN 에서 **a mask IoU scoring module** 이 제안됐었다.
  3. basis head에서 global view를 바라봤었다면, 이 과정을 통해서,  an instance-level(local view)을 바라보게 된다. 
  4. mask IoU score = the IoU + boundary quality 로 분리해 고려했다.
  5. S_boundary 정의 : Boundary score 
     - a Dice metric [13] 을 차용했다. 
     - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210119181451921.png?raw=tru" alt="image-20210119181451921" style="zoom:80%;" />
     - i는 i번째 pixel을 의미하고, epsilon은 분모가 0이 되는 것을 막는다.
  6. <u>Scoring head(위 오른쪽 묘듈)</u>
     - **S_boundary와 S_IOU** 를 학습의 Loss 함수로 사용한다.
     - ~~Input은 concatenation of \[predicted mask (M_pred), boundary (B_pred), RoI-pooled FPN features (F_RoI)\]~~
     - **결론 및 효과** : only for test-time. 학습을 하는 동안 이 score도 추측할 수 있게 만들어 놓으면, test할때 확율값이 얼마인지 확인가능하기도 하고, 학습 성능이 좋아지는 효과도 볼 수 있다.
  7. Score definition **at inference**
     -  Object Detection 모델도 confidence score를 예측하듯이, Mask Score를 예측하는 부분도 만들었다. ~~GT가 뭐고 아래를 어떻게 사용하는지는 나중에 논문이나, 코드 참고하자.~~
     -  <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210119183330139.png?raw=tru" alt="image-20210119183330139"  />



---

## 5. Experiment - Ablation study

- **우리 모델에서의 main 'components'** 
  1. Holistic Boundary basis
  2. Boundary-aware mask scoring  (Boundary_S + Mask_S)
- **Basis head** design choices
  - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210119184502140.png?raw=tru" alt="image-20210119184502140" style="zoom:%;" />
  - ~~boundary supervision을 globally하고, the boundary supervision_loss and its prediction를 모두 사용하는 (e)에서 가장 성능이 좋았다.~~ 
  - ~~마지막 단의 색갈 channel은 original basis를 의미하고, 특히 Red는 the additional image boundary basis를 의미한다.~~ 







---

---

#  B2Inst WACV2021 영상

![1](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/theDevil_boundary/theDevil_boundary_presentation%20P1.png?raw=true)

![1](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/theDevil_boundary/theDevil_boundary_presentation%20P2.png?raw=true)

![1](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/theDevil_boundary/theDevil_boundary_presentation%20P3.png?raw=true)

![1](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/theDevil_boundary/theDevil_boundary_presentation%20P4.png?raw=true)

![1](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/theDevil_boundary/theDevil_boundary_presentation%20P5.png?raw=true)

![1](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/theDevil_boundary/theDevil_boundary_presentation%20P6.png?raw=true)



























