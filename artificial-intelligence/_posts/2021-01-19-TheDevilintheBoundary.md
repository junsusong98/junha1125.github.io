---
layout: post
title: 【segmentation】The Devil Boundary for Instance Segmentation w/ advice
---

- **논문** : [The Devil is in the Boundary: Exploiting Boundary Representation for Basis-based Instance Segmentation](https://arxiv.org/abs/2011.13241)

- **동영상** : [https://www.youtube.com/watch?v=XvLo5WrtHu0](https://www.youtube.com/watch?v=XvLo5WrtHu0) -> why, how, what in the past 에 대한 내용들이 잘 담겨 있으니 꼭 참고.

- **저자** : Myungchul Kim, Sanghyun Woo, Dahun Kim, In So Kweon 

- **읽는 배경** : 현재 Domain adapation, Self-supervise, Transformer 등에 관심이 가지만, 그래도 가장 구체적이며 많은 지식을 가지고 있는, Segmentation이나 Object Detection과 관련된 이 논문을 먼저 읽어보고 싶었다. 같은 랩 석사 선배님이 적으신 논문으로써, 내가 1년 안에 이런 논문을 만들어야 한다는 마음 가짐으로 논문을 읽어보려고 한다. 

- **읽으면서 생각할 포인트** : Reference를 어떻게 추가했나? 실험은 어떤 것을 어떻게 하였나? Relative work는 어느정도로 작성하였나? 과거 지식은 어느 정도가 필요한가? 코드 개발은 어떻게 하였나? (공개된 코드 없으면 선배님한테 여쭤보기)

- **느낀점**  

  1. 논문 안에는 핵심 내용(global 사용, score 사용) 등이 있는데, 최근 논문들의 핵심 내용만 쏙쏙 캐치해서 그것들을 잘~융합해서 개발이 되었다. -> 이런 논문 작성 방식도 추구하자. 
  2. 논문 많이 읽어야 겠다... 완벽하게 이해는 안되고, 60% 정도만 이해가 간다. 필수적으로 읽어야 하는 논문 몇가지만 읽고 나면 다 이해할 수 도 있을 듯 하다. 지금은 전체다 이해가 안된다고 해도 좌절하지 말자.

  3. 정말 많은 노력이 보였다. 내가 과연 이정도를 할 수 있을까? 라는 생각이 들었지만 딱 한달 동안 이와 관련된 매일 논문 1개씩 읽는다면, 잘하면 좀더 창의력과 실험을 가미해서 더 높은 성능을 내는 모델을 만들 수 있지 않을까? 하는 생각은 든다. 따라서 하나의 관심 분야를 가지고 그 분야의 논문 20개는 읽어야 그 쪽은 조금 안다고 할 수 있다. 

  4. 만약 Segmentation을 계속 하고 싶다면, 아래의 '필수 논문'을 차례대로 읽도록 하자. 

  5. **아래에 요약을 정말 쓰레기 처럼 했다. 이때는 Behnam Neyshabur's advice 를 모르고 요약했다. 이제부터 잘 요약하면 된다.** 

   

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
  4. 왜 이런 논문이 나왔지? 지금까지의 계보는 어떻고, 그런 계보가 나온 이유가 무엇이지? 어떤 개념과 어떤 문제, 어떤 정의에 의심을 가지고 시작된 연구이지? 라는 생각을 가지고 논문을 읽고 공부하는 것이 굉장이 중요하다. 예를 들어서 mask score이 나온 이유는 뭐지? boundary score가 사실은 더 중요한거 아닐까? mask-rcnn에서 masking이 왜 잘되지? BB가 이미 잘 만들어지기 때문에? 이걸 없애야 하지 않을까?? 그렇다면 어떻게 없애야지? 이 loss, 이 score가 과연 맞는 건가? 이런 **의심, 질문, 반박**을 찾아 논문을 읽고, 나도 이런 마음을 가지고 생각하는 것이 굉장히 중요하다.
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
  - Single shot instance segmentation with point representation - global-area-based methods - 2020
  - a Dice metric [13] 
  - CondInst [37](https://arxiv.org/pdf/2003.05664.pdf)
  - basis 개념 [4, 5, 38, 6, 37] 중에 적당한거 하나



## 0. Abstract

- <u>the present</u> : single-stage instance segmentation : 실시간성과 정확한 scene understanding.
- <u>ours</u> 
  - Boundary Basis based Instance Segmentation (**B2-In-st**).
  - this Network **learn a <u>globa</u>l boundary representation**.  global-mask-based methods 에서 boundary에 대한 개념도 추가.
  - new unified **quality** **measure** (새로운 통합 측정법) of both mask and boundary = new **score** for the per-instance predictions.
  -  instance segmentation에서,  **state-of-the art** methods on the COCO dataset.



## 1. Introduction

- the present and problems
  - most instance segmentation 의 문제점 
    1. two-stages-instance-segmentation은 first-stage's bounding box에 mask결과값이 많이 의존된다(step-wise).  
    2. 성능 구림(region-specific information / ROI-pooled features 때문에) 이 두가지 문제를 아래의 것들로 해결했다.
  - 최근까지 자주 사용되고 있는 [4, 5, 38, 6, 37] 
    1. **basis framework** : 'global image-level information' 을 결합해서 사용 
    2. 하지만 정확한 a supervised way로 'global mask representations'를 학습하진 않는다. 
    3. last mask output signal에 의해서 학습 되므로(?)
  - boundary 관점에서 집중하는 최근 논문
    - **Boundary-preserving Mask RCNN (2-stage)** : **a boundary prediction head** in 'mask head' of the two-stage Mask RCNN
- <u>우리의 방법</u>
  - a boundary prediction head + the recent **single-stage** instance segmentation = **the RoI-wise head of Boundary-preserving** 
  - **holistic image-level** instance boundaries(global mask representations) be learned -> distinct advantages
  - Use **the boundary ground-truths** 
  - Learn/Use **a novel instance-wise boundary-aware mask score**
- <u>**To sum up** (내용은 비슷하지만, 다른 단어, 다른 문장 사용)</u> ⭐⭐​
  - both on the **global** and the **local** view, the boundary representation 
  - the boundary-aware mask **score** : both the mask and boundary quality simultaneously
  - **single-stage** instance segmentation methods
  - **Blend-Mask**(strongest single-stage instance segmentation methods)를 보완하여 만듬



## 2. Related Work

- two-stage

  1. Mask RCNN : RoIAlign 
  2. PANet : FPN+Mask 
  3. Mask scoring RCNN (읽어야함): alignment between the confidence score and localization accuracy of predicted masks

  - 문제점 :  (1) RoI-wise feature pooling에 의한 (잠재적) 문제 발생 (2) quite slow

- Single-stage 

  - the present and probelms : 
    - Instead of assigning ROI proposals, **pixel-wise predictions of cues** such as (a)directional vectors (b)pairwise affinity (c)watershed energy (d)embedding learning (=  **local-area-based methods,** 각각에 대한 설명은 각 paper 참고) -> and then **grouping **pixel-wise predictions for object instances  -> not good ....
    - **global-area-based** methods 탄생 [6, 4, 38] : 
      1. Generate intermediate FCN feature maps, called ‘**basis**’. 
      2. Assemble the extracted basis features.
  - 뒤에서 더 다룰 예정.

- **Boundary learning** for instance segmentation

  - the past and problems
  - CASENet : sementic segmentation.  category-aware boundary detection.
    - InstanceCut : instance-level.
      - 이 둘은 expensive! (super-pixel extractions and grouping computations 때문에)
  - the present and problems
    - Zimmermann et al.[47]:  boundary agreement head,  **auxiliary edge loss**.
    - Boundary-preserving mask r-cnn[11] : **instance-level boundaries head** ([47]에서 업그레이드)
      - 이 둘은 two-stage methods(RoI-wise)! 이다. **ROI는 local! Lack a holistic view** 



## 3. Background (single-stage)

- **Basis**-based instance segmentation의 시작 : (= global-area-based)
  1. FCN으로 **a set of basis mask representations(?)** 를 생성한다. 
  2. (1을 하는 동안 동시에) Detected Box를 찾고, 그것으로 **instance-level parameters(= instance-specific information)**를 예측한다.
  3. (1과 2를 결합해서) instance segmentation을 수행한다. 
- 위의 pipeline을 따르는 모델들
  1. **YOLACT** :  32 global base, the according instance-specific scalar coefficients, a linear combination among the bases.  --->  cause [rich instance features **vs** effective assembly methods\]
  2. improved the assembly parameters : **BlendMask** [6], SipMask [5], and **CenterMask** [38] 
  3. instance-specific representation : **CondInst** [37] (Key word: dynamic filter weights, sliding window, the predicted convolution weight filters)
- our step ⭐⭐​
  1. **under-explored boundary information** 
  2. **holistic boundaries** (the global basis representation)
  3. **a boundary-aware mask score** (about the mask and boundary quality for inference)



## 4. Boundary Representation

- **B2Inst**
  1. **backbone** for feature extraction
  2. instance-specific detection head (**Detection head**)
  3. the global basis head,  the global image information (**Basis head**)
  4. **mask scoring head**
  5. (2번)과 (3번)을 **결합**해서, final masks 예측.
  6. BlendMask instantiation (boundary basis)

<img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210119165307132.png?raw=tru" alt="image-20210119165307132" style="zoom:80%;" />

- Details
  1. Standard **basis head**(그림 중간 아래)
     - FPN feature를 받고, a set of K(=#Bases) global basis features를 생성한다 
     - 4개의 3x3 conv 통과 + upsampling  + reduce the channel-size(#Bases)
     - The previous basis head is supervised by the last mask loss only.
  2. image boundary(Missing piece 그림에 없음)
     - two-stage에서는 RoI-wise boundary information을 이용해서, 성능향상을 이뤘다.
     - 하지만 우리는, Learn **a holistic boundary of all instances** in a scene (instance 하나하나 아니라)
     - Overlapping objects and complex shapes문제 에서 좋은 성능 나옴.
     - the boundary supervision 은 어렵지 않다. mask annotations를 그냥 이용하면 되므로.
       - Boundary ground-truths: Laplacian operator to generate soft boundaries from the binary mask ground-truths.
  3. Objective function
     - 1) binary cross-entropy loss   
     - 2) dice loss(다룰 예정) 
     - 3) boundary loss 
  4. **Detection Head**
     - Basic head와 병렬로, instance-level parameters 들을 추측한다. 
     - 특히 여기서 나오는 attention map feature는 basic head의 결과와 결합되어 **Boundary Basic**(BlendMask 참고)이 이뤄진다.
- Boundary-aware mask scoring
  1. Mask scoring R-CNN [20] proposes **a mask IoU scoring module** instead of the classification score.
  2. Mask 결과를 향상시킬 수 있다. 모델은 이 값이 커지도록 학습을 할테니까. 즉 GT와 최대한 비슷해지려고 하는 back-propagation이 여기서 된다고 할 수 있다. 
  3. 이 과정을 통해서,  an instance-level(local view)로 학습이 이뤄지고, basis head로 global view를 바라본다.
  4. 저자는 mask IoU score = the IoU + boundary quality 로 나누어 생각했다.
  5. S_boundary 정의 : Boundary score 
     - a Dice metric [13] 을 차용했다. 
     - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210119181451921.png?raw=tru" alt="image-20210119181451921" style="zoom:80%;" />
     - i는 i번째 pixel을 의미하고, epsilon은 분모가 0이 되는 것을 막는다.
  6. <u>Scoring head(위 오른쪽 묘듈)</u>
     - **S_boundary와 S_IOU**는 GT와 prediction결과를 비교해서 쉽게 계산 할 수 있다. 이것이 높게 나오도록 boundary와 Mask결과를 찾는 모델의 학습.이 이뤄지겠지만
     - **이 값을 예측하는 모듈**도 학습시켰다. (Object Detection 모델도 confidence score를 예측하듯이)
     - Input은 concatenation of \[predicted mask (M_pred), boundary (B_pred), RoI-pooled FPN features (F_RoI)\]
     - **결론 및 효과** : only for test-time. 학습을 하는 동안 이 score도 추측할 수 있게 만들어 놓으면, test할때 확율값이 얼마인지 확인가능하기도 하고, 학습 성능이 좋아지는 효과도 볼 수 있다.
  7. Score definition **at inference**
     -  <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210119183330139.png?raw=tru" alt="image-20210119183330139" style="zoom:80%;" />



## 5. Experiment - Ablation study

- The main 'components' of our framework design 
  1. Holistic Boundary basis
  2. Boundary-aware mask scoring  (Boundary_S + Mask_S)
- **Basis head** design choices
  - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210119184502140.png?raw=tru" alt="image-20210119184502140" style="zoom:67%;" />
  - boundary supervision을 globally하고, the boundary supervision_loss and its prediction를 모두 사용하는 (e)에서 가장 성능이 좋았다. 
  - 마지막 단의 색갈 channel은 original basis를 의미하고, 특히 Red는 the additional image boundary basis를 의미한다. 

































