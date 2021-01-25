---
layout: post
title: 【Domain】Adversarial Discriminative Domain Adaptation = ADDA 
---

**논문** : [Adversarial Discriminative Domain Adaptation - y2017,c1875](https://arxiv.org/pdf/1702.05464.pdf)

**분류** : Unsupervised Domain Adaptation

**저자** : Eric Tzeng, Judy Hoffman, Kate Saenko (California, Stanford, Boston University)

**읽는 배경** : (citation step1) Open Componunt Domain Adaptation에서 Equ (1), (2) \[the domain-confusion loss\]가 이해가 안되서 읽는 논문.  근데 introduction읽다가 이해해 버렸다. 좀만 더 읽어보자.

**읽으면서 생각할 포인트** : 핵심만 읽자. 모든 것을 다 읽으려고 하지 말자. 영어계 저자는 어떻게 썼는지 한번 확인해 보자.



**질문**

1. 어떤 논문을 찾아 읽지? 
    -> 다 읽어 뭘 고민해. 최대한 다 읽어. 다 읽고 핵심만 쏙쏙 알아놔 



**느낀점**  

1. 이해 50프로 밖에 못한것 같다. 
2. 이해를 많이 못한 이유. 
   - 굳이 이해하고 싶지 않다. 너무 옛날 논문이다. 과연 이게 정석인지는 모르겠다. 하지만 미래의 논문들은 정말 오래 전 과거의 지식들과 융합해서 더 좋은 결과를 내는 스토리의 논문들도 있다. 그러니 이런 생각 너무 갖지 말자
   - 내가 이해한게 정말 맞는지 모르겠다. 뭔가 맞는 것 같기도 하고, 아닌것 같기도 하다. 확실히 코드를 봐야 정확히 알 수 잇는 것 같다. 
3. 이 논문을 읽으면서, "**이해안되도 아가리 닥치고 일단 Experience까지 미친듯이 읽어보자.** 그리고 다시 과거에 이해는 안되지만 대충이라도 요약했던 내용을, 다시 읽어보면! 이해가 될 가능성이 높다." 
4. 그래도 이 논문을 통해서 Adversarial 접근이 GAN의 generator와 discriminator만을 두고 얘기하는게 아니라는 것을 깨달았다. 하나는 '이러한 목적'을 위해 학습한다면, 다른 한쪽에서 '반대의 목적'을 위한 Loss로 학습하게 만드는 것들, 전부가 Adversarial learning이라는 것을 깨달았다. 
5. 확실한 딥러닝 흐름을 공부하는 방법을 아직도 잘 모르겠다. 모르는 거에 너무 집착하지 말고, 새로 나오는 기술들에 집중하며, 전체적인 흐름을 느끼고, 새로운 논문이 나온 이유에 대해서도 집중하는 것. 이 딥러닝을 공부하는 방법이라고 하지만 그래도 그게 어느정도 인지 모르겠다. 
6. **하지만 나는 믿는다. 그냥 아가리 닥치고 계속 읽고 정리하고 많은 논문을 읽어가다 보면 언젠간. 언젠간. 많은 지식이 쌓여 있고, 누구에게 뒤쳐지지 않는 나를 발견할 수 있을거라는 사실을. 그니까 그냥 해보자. 좌절하지말고 재미있게 흥미롭게.**



**<u>다 읽은 후, 필수로 읽어야 겠다고 생각이 든 논문</u>**




# 0. Abstract

- the present & problem
  - Adversarial learning(approaches) 사용 및 효과 
    - generate diverse domain data + improve recognition despite target domain.
    - reduce the difference, Improve generalization performance.
  - generative approaches 문제점 - smaller shifts만 가능.
  - discriminative approaches 문제점 -   larger domain shifts, But  **tied weights(??), GAN-based loss 사용안함.**
- Ours ⭐⭐
  - 지금까지의 접근방법들을 잘 융합하고, 적절히 변형한다. 
  - Use **(1) discriminative modeling(base models) (2) untied weight sharing, (3) GAN loss(adversarial loss)**
  - 즉 우리 것은 이것이다. general(=generalized, optimized) framework of discriminative modeling(=adversarial adaptation) = ADDA
  - SOTA -  digit classification,  object classification



# 1. Introduction

- the past problem & effort
  - dataset bias + domain shift 문제해결은 일반적으로  fine-tune. But labeled-data is not enough.
  - 서로 다른  feature space를 mapping 시켜주는 deep neural transformations 방법도 존재한다. 그때 이런 방법을 사용했다. maximum mean discrepancy [5, 6] or correlation distances [7, 8]  또는 
  - the source representation(source를 encoding한 후 나오는 feature)로 부터 target domain을 reconstruct하는 방법도 있다.[9] (encoding + decoding의 결과가 target data모양이 되도록. 그렇게 만든 data로 classifier 재 학습??)
- the present
  - Adversarial adaptation : domain discriminator에서 adversarial objective(Loss)를 사용하여, domain discrepancy를 최소화 했다. 
  - generative adversarial learning [10] : generator(이미지 생성) discriminator(generated image, real image 구별) 를 사용해서 network가 어떤 domain인지 판단하지 못하게 만든다. [10,11,12,13]  이것들의 방식은 조금씩 다르다. 예를 들어, 어디에 generator를 위치시키고, loss는 무엇이고, weight share for source&target을 하는지 안하는지 등등
- Ours
  - 우리는 위 방식들을 최적한 novel unified framework를 만들어 냈다. - 위 Ours의 (1)(2)(3)
  - generative modeling 은 필수적인게 아니라, discriminative representation를 학습하는게 우선이라고 판단했다. 
  - <img src="C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210122185525171.png" alt="image-20210122185525171" style="zoom:67%;" />
  - ADDA는 (위 사진 참조) 
    1. Learn discriminative representation using the labels(????) in the source domain.
    2. Learn  a separate encoding through a domain-adversarial loss.



# 2. Related work

- MiniMizing Difference between feature distributions.[MMD사용모델들]
  1. MMD[3] : Computes **the norm of the difference between two domain means**.
  2. DDC[5] : MMD in addition to the regular classification loss for both discriminative and domain invariant.
  3. The Deep Adaptation Network[6] :  effectively matching higher order statistics of the two distributions
  4. CORAL [8] : match the mean and covariance of the two distributions
- Using **adversarial loss** to minimize domain shift. & learning a representation not being able to distinguish between domains(어떤 domain이든 공통된 feature extractor 제작) - 2015
  1. [12] : a domain classifier and  a domain confusion loss
  2. ReverseGrad[11] : the loss of the domain classifier by reversing its gradients
  3. DRCN[9] : [11]과 같은 방식 + learn to reconstruct target domain images
- GAN - for generating - 2015
  1.  G : capture distribution. D : distinguishes. (Generative Adversarial Network)  
  2.  BiGAN [14] : learn the inverse mapping(??) from image to latent space and also learn useful features for image classification task.
  3. CGAN [15] : generate "a distribution vector" conditional on image features. and G and D receive the additional vector.
- GAN for domain transfer problem - 2013
  1. CoGAN [13] : generate both source and target images respectively. 이로써 a domain invariant feature space 를 만들어 낼 수 있었다.  discriminator output 윗단에 classifier layer를 추가해서 학습을 시켰다. 이게 좋은 성과를 냈지만, source와 target간의 domain 차이가 심하다면 사용하기 어려운 방법이다. 하지만 쓸데없이 generator를 사용했다고 여겨진다.
- Ours (위 figure 참조))
  1. **image distribution(분포, 확률분포, generator)는 필수적인게 아니다**.
  2. 진짜 중요한 것은, **discriminative approach** 이다.
- 정리표 : <img src="C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210123225939714.png" alt="image-20210123225939714" style="zoom:90%;" />





## 3. Generalized adversarial adaptation - related work

- ![image-20210123225514370](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210123225514370.png)
  1. Source CNN 학습시키기.
  2. Adversarial adaptation 수행. Target CNN을 학습.
  3. Discriminator는 cannot reliably predict their domain label 하게 만든다. 
  4. test 할 때, target encoder to the shared feature space(?) 그리고 source classifier를 사용한다. 
  5. 점선은 Fixed network parameters를 의미한다.

- 일반적인 adversarial adaptation 의 formula ⭐⭐
  - <img src="C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210125141320232.png" alt="image-20210125141320232" style="zoom:80%;" />
- 3.1. Source and target mappings (Mapping - Ms,Mt는 어떻게 설정해야 하는가?)
  - 목적 : mapping 신경망이 source에서든 target에서든 잘 동작하게 만들기. 각각을 위한 mapping 신경망이 최대한 가깝게(비슷하게) 만들기. source에서든 target에서든 좋은 classification 성능을 내기
  - 과거의 방법 : mapping constrain = target과 source를 위한 mapping. Ms,Mt = feature extractor = network parameter sharing 
  -  Ours : partial alignment = partially shared weights 
- 3.2. Adversarial losses (위의 Loss_adv_M은 무엇으로 해야하는가?)
  - [16] ![image-20210125141617468](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210125141617468.png) 을 사용하기도 했지만, 이 방법에서 Discriminator가 빨리 수렴하기 때문에, (같이 적절하게 학습된 후 동시에 수렴되야 하는데..) 문제가 있다.
  - **GAN loss function** [17] ![image-20210125141823643](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210125141823643.png) provides stronger gradients to the target mapping.(Mt는 Xt를 잘 classification 하도록 학습되면서도, D가 잘 못 판단하게 만들게끔 학습 된다. ) -> 문제점 : oscillation. 둘다 너무 수렴하지 않음. D가 괜찮아지려면 M이 망하고, M이 괜찮아 지려만 D가 망한다. 
  - [12] : <img src="C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210125142625114.png" alt="image-20210125142625114" style="zoom: 80%;" />  
    target이 들어오면, D가 잘못 판단하게끔 Mt가 학습되면서도, D가 잘 판단하게끔 Mt가 학습된다. 반대로 source가 들어오면, 또 D가 잘못 판단하게끔 Ms가 학습되면서도, D가 잘 판단하게끔 만드는 항도 있다. 
- 이러한 고민들이 계속 있었다. Ours의 결론은 위의 정리표 참조.<img src="C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210125143353867.png" alt="image-20210125143353867" style="zoom:90%;" />



## 4. Adversarial discriminative domain adaptation

- 사용한 <u>objective function</u>  ⭐⭐⭐⭐
  - ![image-20210125144836013](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210125144836013.png)
- 최종 모델 학습 과정 - 헷갈리는 내용은 다시 <u>논문 5 page</u> ⭐⭐참조. 
  - ![image-20210125145015365](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210125145015365.png)



# 5. Experiments

- 실험을 통해서 왜 논문에서 선택한 위의 objective function이 적절한 function이었는지를 말해준다.