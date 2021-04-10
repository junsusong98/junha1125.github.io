---
layout: post
title: 【Domain】Adversarial Discriminative Domain Adaptation = ADDA 
---

- **논문** : [Adversarial Discriminative Domain Adaptation - y2017,c1875](https://arxiv.org/pdf/1702.05464.pdf)
- **분류** : Unsupervised Domain Adaptation
- **저자** : Eric Tzeng, Judy Hoffman, Kate Saenko (California, Stanford, Boston University)
- **읽는 배경** : (citation step1) Open Componunt Domain Adaptation에서 Equ (1), (2) \[the domain-confusion loss\]가 이해가 안되서 읽는 논문.  근데 introduction읽다가 이해해 버렸다. 좀만 더 읽어보자.
- **읽으면서 생각할 포인트** : 핵심만 읽자. 모든 것을 다 읽으려고 하지 말자. 영어계 저자는 어떻게 썼는지 한번 확인해 보자.



# 느낀점

1. 이해 50프로 밖에 못한것 같다. 
2. ~~이 논문을 읽으면서, "**이해안되도 아가리 닥치고 일단 Experience까지 미친듯이 읽어보자.** 그리고 다시 과거에 이해는 안되지만 대충이라도 요약했던 내용을, 다시 읽어보면! 이해가 될 가능성이 높다."~~ 
3. 그래도 **이 논문을 통해서 Adversarial 접근이 GAN의 generator와 discriminator만을 두고 얘기하는게 아니라는 것을 깨달았다. 하나는 '이러한 목적'을 위해 학습한다면, 다른 한쪽에서 '반대의 목적'을 위한 Loss로 학습하게 만드는 것들, 전부가 Adversarial learning이라는 것을 깨달았다.** 
4. **하지만 나는 믿는다. 그냥 아가리 닥치고 계속 읽고 정리하고 많은 논문을 읽어가다 보면 언젠간. 언젠간. 많은 지식이 쌓여 있고, 누구에게 뒤쳐지지 않는 나를 발견할 수 있을거라는 사실을. 그니까 그냥 해보자. 좌절하지말고 재미있게 흥미롭게.** 않는 나를 발견할 수 있을거라는 사실을. 그니까 그냥 해보자. 좌절하지말고 재미있게 흥미롭게.**
5. **<u>(3)번까지 relative work다 읽느라 지쳐서 핵심적인 부분은 거의 이해도 안하고 대충대충 넘겼다. 진짜 중요한 부분이 어딘지 생각해봐라. 초반에 이해안되는거 가지고 붙잡고 늘어지기 보다는, 일단 이 논문이 제시하는 model이 구체적으로 뭔지부터 읽는 것도 좋았겠다.</u>**
6. <u>헛소리 이해하려고, 핵심적인 4번 내용은 이해하지도 않았다. 솔직히 여기 있는 relative work를 언제 다시 볼 줄 알고, 이렇게 깔끔하게 정리해 놨냐. 정말 필요하면 이 직접 논문으로 relative work를 다시 읽었겠지. 그러니 **핵심 먼저 파악하고!! 쓸대 없는 잡소리 이해는 나중에 하자.**</u>
7. **<u>별거도 아닌거 삐까번쩍하게도 적어놨다... 이래서 아래로 최대한 빨리 내려가서 구체적인 핵심먼저 파악해야한다.</u>**




# 0. Abstract

- **현재 기술들과 문재점** 
  - Adversarial learning(approaches) 사용 및 효과 
    - generate diverse domain data를 한 후에 improve recognition despite target domain을 수행한다.
    - 그렇게 해서 reduce the difference하고 Improve generalization performance한다.
  - 하지만 이런 generative approaches의 문제점은 - smaller shifts만 가능하다는 것이다.
  - discriminative approaches 문제점 -   larger domain shifts가 가증하지만,  **tied(=sharing) weights, GAN-based loss를 사용하지 않는다.**
- **Ours **
  - 지금까지의 접근방법들을 잘 융합하고, 적절히 변형한다. 
  - Use (별거도 아닌거 삐까번쩍하게도 적어놨다... 이래서 아래로 최대한 빨리 내려가서 구체적인 핵심먼저 파악해야한다.)
    - **(1) discriminative modeling(base models) **
    - **(2) untied weight sharing, **
    - **(3) GAN loss(adversarial loss)**
  - 즉 우리 것은 이것이다. general(=generalized, optimized) framework of discriminative modeling(=adversarial adaptation) = ADDA
  - SOTA 달성했다. -  digit classification,  object classification
  - **논문 핵심** : (Related work) Adversarial 을 사용하는 방법론들은 항상 같은 고충을 격는다. "Discriminator가 잘 학습이 되지 않는다든지, Discriminator를 사용해서 Adversarial 관점을 적용해도 원한는데로 Model이 target data까지 섭렵하는 모델로 변하지 않는다던지 등등 **원하는데로 학습이 잘 이뤄지지 않는 문제점**" 을 격는다. 이 논문은 많은 문제점을 고려해서 가장 최적의 Discriminator 공식과 Adversarial 공식을 완성했다고 주장한다.



# 1. Introduction

- **과거의 방법론들**
  - dataset bias(source만 데이터 많음 target없음) + domain shift 문제해결은 일반적으로  fine-tune으로 했었다. But labeled-data is not enough.
  - 서로 다른  feature space를 mapping 시켜주는 deep neural transformations 방법도 존재한다. 그때 이런 방법을 사용했다. maximum mean discrepancy [5, 6] or correlation distances [7, 8]  또는 
  - the source representation(source를 encoding한 후 나오는 feature)를 decoder에 집어넣어서, target domain을 reconstruct하는 방법도 있다.[9] (encoding + decoding의 결과가 target data모양이 되도록. 그렇게 만든 data로 classifier 재 학습??)
- **현재의 기술들**
  - Adversarial adaptation : domain discriminator에서 adversarial objective(Loss)를 사용하여, domain discrepancy를 최소화 했다. 
  - generative adversarial learning [10] : generator(이미지 생성) discriminator(generated image, real image 구별) 를 사용해서 network가 어떤 domain인지 판단하지 못하게 만든다. [10,11,12,13]  이것들의 방식은 조금씩 다르다. 예를 들어, 어디에 generator를 위치시키고, loss는 무엇이고, weight share for source&target을 하는지 안하는지 등등
- **Ours**
  - 위 Ours의 (1)(2)(3)
  -  discriminative representation를 학습하는게 우선이다. 
  - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210122185525171.png?raw=tru" alt="image-20210122185525171" style="zoom:67%;" />
  - ~~ADDA 간략히 정리하자면...~~
    1. ~~Learn discriminative representation (using the labels in the source domain). -> 그냥 Source classifier 학습시키기.~~ 
    2. ~~Learn  a separate encoding through a domain-adversarial loss. (domain discriminator 학습 및 target CNN 학습시키기)~~



# 2. Related work

### **<u>(이렇게 자세하게 적는게 무슨 의미지? 반성하자. 어차피 머리속에 남은것도 없고, 다시 읽어도 뭔소린지 모르고 머리에도 안남잖아. 시간버린거야.)</u>**

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
- 정리표 : <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210123225939714.png?raw=tru" alt="image-20210123225939714" style="zoom:90%;" />





# 3. Generalized adversarial adaptation - related work

- 우리의 학습 순서 요약
  
  - ![image-20210123225514370](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210123225514370.png?raw=tru)
  
  1. Source CNN 학습시키기.
  2. Adversarial adaptation 수행. 
  3. Target CNN을 학습.
- 먼저 학습시켰던 Discriminator를 사용해서,  cannot reliably predict their domain label 하게 만듬으로써 Target CNN을 학습시킨다.
- test 할 때, target encoder to the shared feature space(?) 그리고 source classifier(targetCNN도 원래는 Source CNN을 기반으로 하는 모델이므로)를 사용한다.
- 점선은 Fixed network parameters를 의미한다.
- (Related work) Adversarial 을 사용하는 방법론들은 항상 같은 고충을 격는다. "Discriminator가 잘 학습이 되지 않는다든지, Discriminator를 사용해서 Adversarial 관점을 적용해도 원한는데로 Model이 target data까지 섭렵하는 모델로 변하지 않는다던지 등등 **원하는데로 학습이 잘 이뤄지지 않는 문제점**" 을 격는다. 이 논문은 많은 문제점을 고려해서 가장 최적의 Discriminator 공식과 Adversarial 공식을 완성했다고 주장한다.
  1. 일반적인 adversarial adaptation 의 formula 

    - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210125141320232.png?raw=tru" alt="image-20210125141320232" style="zoom:100%;" />

  2. **3.1. Source and target mappings (Mapping - Ms,Mt는 어떻게 설정해야 하는가?)**
    - 목적 : mapping 신경망이 source에서든 target에서든 잘 동작하게 만들기. 각각을 위한 mapping 신경망이 최대한 가깝게(비슷하게) 만들기. source에서든 target에서든 좋은 classification 성능을 내기
    - 과거의 방법 : mapping constrain = target과 source를 위한 mapping. Ms,Mt = feature extractor = network parameter sharing 
    -  Ours : partial alignment = partially shared weights 

  3. **3.2. Adversarial losses (위의 Loss_adv_M은 무엇으로 해야하는가?)**

    - [16] ![image-20210125141617468](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210125141617468.png?raw=tru) 을 사용하기도 했지만, 이 방법에서 Discriminator가 빨리 수렴하기 때문에, (같이 적절하게 학습된 후 동시에 수렴되야 하는데..) 문제가 있다.
    - **GAN loss function** [17] ![image-20210125141823643](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210125141823643.png?raw=tru) provides stronger gradients to the target mapping.(Mt는 Xt를 잘 classification 하도록 학습되면서도, D가 잘 못 판단하게 만들게끔 학습 된다. ) -> 문제점 : oscillation. 둘다 너무 수렴하지 않음. D가 괜찮아지려면 M이 망하고, M이 괜찮아 지려만 D가 망한다. 
    - [12] : <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210125142625114.png?raw=tru" alt="image-20210125142625114" style="zoom: 80%;" />  
      target이 들어오면, D가 잘못 판단하게끔 Mt가 학습되면서도, D가 잘 판단하게끔 Mt가 학습된다. 반대로 source가 들어오면, 또 D가 잘못 판단하게끔 Ms가 학습되면서도, D가 잘 판단하게끔 만드는 항도 있다. 

  4. 이러한 고민들이 계속 있었다. Ours의 결론은 위의 정리표 참조.

    - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210125143353867.png?raw=tru" alt="image-20210125143353867" style="zoom:90%;" />



# 4. Adversarial discriminative domain adaptation 

- 사용한 <u>objective function</u>  ⭐⭐   
  ![image-20210125144836013](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210125144836013.png?raw=tru)
  1. 맨위 수식 : Source Dataset의 Classification을 잘하는 모델 탄생시킨다. 
  2. 최적의 Discriminator 공식 : Target이 들어오면 0을 out하고, Source가 들어오면 1을 out하도록 하는 Discriminator를 만들기 위한 Loss 함수이다. 
  3. 심플하게 Source 신경망은 고정시킨다. Target이미지를 넣어서 나오는 Out이 Discriminator가 1이라고 잘못 예측하게 만드는 M_t만 학습시킨다. Ms는 건들지도 않고 source가 들어갔을때, discriminator가 0이라고 잘못 예측하게 만든는 작업 또한 하지 않는다.
- 최종 모델 학습 과정
  - ![image-20210125145015365](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210125145015365.png?raw=tru)
  - 헷갈리는 내용은 다시 <u>논문 5 page</u> ⭐⭐참조. 



# 5. Experiments

- 실험을 통해서 왜 논문에서 선택한 위의 objective function이 적절한 function이었는지를 말해준다.