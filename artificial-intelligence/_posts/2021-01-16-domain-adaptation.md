---
layout: post
title: 【Domain-Adaptation】Deep Domain Adaptation Basics
---
Deep Domain Adaptation Basics

- 2021.02.16 전체 복습 후 느낀점 : 정말 이 분야를 제대로 하고 싶다면, 아래의 내용들은 느낌만 알아가는데에 좋다. 더 자세한 내용들은 결국 논문을 찾아서 읽어야 한다.





# Reference (Study order)

1. [https://towardsdatascience.com/deep-domain-adaptation-in-computer-vision-8da398d3167f](https://towardsdatascience.com/deep-domain-adaptation-in-computer-vision-8da398d3167f)
2. [http://www.eecs.umich.edu/eecs/pdfs/events/4142.pdf](http://www.eecs.umich.edu/eecs/pdfs/events/4142.pdf)
3. [https://sudonull.com/post/9686-Overview-of-Deep-Domain-Adaptation-Basic-Methods-Part-1](https://sudonull.com/post/9686-Overview-of-Deep-Domain-Adaptation-Basic-Methods-Part-1)



# 1. Deep Domain Adaptation In Computer Vision

- 이 Reference가 좋은 자료 인지 모르겠다. 애매모호 하기만 하다.

## 1. introduction
  - 특정 신겨망 모델을 NYC dataset로 학습시키고, Manhattan에서는 잘 동작하지만 Paris에서 사용하니 문제 발생! 
  - images from different viewing angles / different lighting conditions 
    - source distribution = source dataset : Pre-train을 위해서 사용했던 데이터셋(의 분포, 분야)
    - targer distribution = source dataset : domain changed 된, dataset.
  - 일반적으로, damain adaptation은 한개 혹은 그 이상의 데이터셋을 사용한다. 



## 2. Domain Adaptation Categories

  - 참고 논문 
    1. Deep Visual Domain Adaptation: A Survey (2018)
    2. A Survey of Unsupervised Deep Domain Adaptation (2019)
    3. 과제의 복잡성은 `labeled/unlabeld data의 양`, `source/target의 차이 정도`로 분류 가능하다.
  - domain adaptation : the task space is the same(해야하는 과제는 같다. maybe Ex. obejct detection, segment. , kinds of labels) But input domain is divergent.
  - 크게 분류
    - homogeneous (동족의) = source : target = 1 : 1 데이터셋
    - heterogeneous (여러 종류,종족의) = compound = source : target =  1 : 3 데이터셋 
  - target data에 따라서 
    - supervised : target data is labeled
    - semi-supervised : labeled and unlabeled
    - self-supervised : No labeled data



## 3. Task Relatedness(상관성, 유사성)

  - task relatedness : source가 task이 얼마나 유사한가?
  - Task Relatedness를 정의(수치화)하는 방법
    1. how close their **parameter vectors**. 
    2. how **same features** 
  - 그럼에도 불구하고, domain adaptation을 사용할 수 있는지 없는지 판단하는 방법은, 직접 학습시키고 테스트 해봐야한다.



## 4. One-Step Domain Adaptation

3가지 basic techniques

1. divergence-based domain adapatation.
2. adversarial-based domain adaptation using GAN, domain-confusion loss.
3. reconstruction using stacked autoencoders.

하나씩 알아가보자.  



## 4-1 Divergence-based Domain Adaptation

- **some divergence criterion (핵심 차이점 기준?) 을 최소화**하고, **domain-invariant feature representation** 을 찾아내는 것(achieve) (domain 변화에도 변하지 않는 feature extractor Ex. 어떤 신호등 모양이든 같은 신호등feature가 나오도록)
- 아래에 3가지 방법이 나오는데, 느낌만 가져가기. 뭔 개소리인지 정확히 모르겠다.
- (1) MMD - Maximum Mean Discrepancy 
  - <img src='https://user-images.githubusercontent.com/46951365/104798356-85f04c80-5809-11eb-8d0d-4fae09915519.png' alt='drawing' width='300' style="zoom: 150%;" />  
  - two-stream architecture는 파라메터 공유하지 않음. 
  - Soft-max, Regularization(Domain-discrepancy) loss를 사용해서, two-architecture가 **similar feature representation(=extractor)**가 되도록 만든다.
- (2) CORAL  - Correlation Alignment
  - <img src='https://user-images.githubusercontent.com/46951365/104798534-f2b81680-580a-11eb-9507-7881188b601d.png' alt='drawing' width='250' style="zoom: 200%;" />
  - (b)처럼 distribution만 맞춘다고 해서 해결되지 못한다. (c)처럼 soure에 target correlation값을 추가함으로써 align시켜준다.
  - align the second-order statistics (correlations) instead of the means
  - [좋은 논문](https://arxiv.org/pdf/1607.01719.pdf) : Using a differentiable CORAL loss.
- (3) CCD - Contrastive Domain Discrepancy
  - label distributions 을 사용한다. (라벨별 확률 분포) by looking at conditional distributions(조건적인 P(확률분포\|특정라벨))
  - 두 데이터의 각 라벨에 대해, 교집합 domain feature를 찾는다. 
  - minimizes(최소화 한다) the intra-class discrepancy, maximizing(최대화 한다) the inter-class discrepancy.
  - [좋은논문](https://arxiv.org/pdf/1901.00976.pdf) : target labels are found by **clustering**. CCD is minimized.
- 이런 방법으로
  1. 이런 방법을 optimal transport 라고 한다. 
  2. 두 데이터 간의 feature and label distributions 가 서로 비슷해지게 만들거나,
  3. **두 architecture(extractor, representation)** 간의 차이가 줄어들게 만든다.



## 4-2 adversarial-based domain adaptation

- **source domain에 관련된 인위적인 target data를 만들고, 이 데이터를 사용해서 target network를 학습시킨다. 그리고 진짜 target data를 network에 넣어서 결과를 확인해 본다.**
- (1) CoGAN - source와 연관성 있는 target data 생성
  - <img src='https://user-images.githubusercontent.com/46951365/104799034-0feee400-580f-11eb-9e81-694aeb61a36a.png' alt='drawing' width='400' style="zoom: 130%;" />
  - 일부 **weight sharing 하는 layer는 domain-invariant feature space extractor**로 변해간다.
- (2) source/target converter network - source와 연관성 있는 target data 생성
  - Pixel-Level Domain Transfer (2016 - citation 232)
  - <img src='https://user-images.githubusercontent.com/46951365/104799089-8b509580-580f-11eb-96c6-9f01c1aaabf7.png' alt='drawing' width='400' style="zoom:130%;" />
  - 2개의 discriminator를 사용한다.
  - 첫번째 discriminator는 source에 의해 생성된 target data가 그럴듯 한지 확인하고.
  - 두번째 discriminator는 생성된 target data와 source의 상관성이 있는지 확인한다.
  - 특히 이 방법은 **unlabeled data in the target domain** 상황에서 사용하기 좋다. 
- (3) Get rid of generators - 어떤 domain에서도 invariable-feature를 추출하는 extractor 제작
  - Unsupervised Domain Adaptation by Backpropagation (2015 - citation 2000)
  - domain confusion loss in addition to the domain classification loss : classificator가 어떤 domain의 data인지 예측하지 못하게 한다. 
  - <img src='https://user-images.githubusercontent.com/46951365/104799241-d919cd80-5810-11eb-945a-194672815009.png' alt='drawing' width='400' style="zoom:130%;" />
  - gradient reversal layer는 the feature distributions를 일치시키기 위해 존재한다.(두 데이터 간의 특징 분포를 일치시키기 위해)
    - 파랑색 부분은 class label를 잘 찾으려고 노력하고
    - 초록색 부분은 domain classifier가 틀리도록 학습되면서(**input이미지에 대해서 어떤 domain에서도 invariable-feature를 추출하는 extractor를 만든다**), class label을 잘 맞추려고 학습된다. 
    - 빨간색은 그대로 domain label을 옳게 classify하도록 학습된다. 
    - Generator & discriminator 구조가 아닌듯, 맞는듯한 신기한 구조를 가지고 있다.



## 4-3. Reconstruction-based Domain Adaptation

- (1) DRCN
  - Deep Reconstruction-Classification Networks for Unsupervised Domain Adaptation (2016 - citation 435)
  - <img src='https://user-images.githubusercontent.com/46951365/104799917-5a725f80-5813-11eb-9229-01abb018a8c6.png' alt='drawing' width='400' style="zoom:150%;" />
  - (i) classification of the source data (ii) reconstruction of the unlabeled target data
  - (i) 나중에 input에 target을 넣어도 잘 classifying 하게 만듬. (ii) reconstruction된 data가 task data와 유사하도록 학습된다. 따라서 초반 layer도 task data에 대한 정보를 함축하도록 만들어 진다.
  - 논문에서는, 위 신경망의 Input=Source Reconstruction=Target을 넣고 먼저 학습시킨다. 그리고 반대로 Input=Target Reconstruction=Source가 되도록 다시 학습시켰다고 한다.
  - 아래와 같은 학습 방법도 가능하다.
  - <img src='https://user-images.githubusercontent.com/46951365/104808630-e04dd500-582a-11eb-867d-4707cfb2d5fd.png' alt='drawing' width='500' style="zoom:150%;" />
- (2) cycle GANs
- (3) conditional GANs
  - encoder-decoder GAN 
  - conditional GANs are used to translate images from one domain to anothe
  - Pix2Pix GAN이 대표적이다. 
  - reference를 주면, 그것을 이용해 ouput을 만드는 GAN을 말한다. 
  - <img src='https://user-images.githubusercontent.com/46951365/104805907-d0c49100-5816-11eb-9f29-26f5a7d12904.png' alt='drawing' width='400' style="zoom:150%;" />



## 5. Conclusion

- Deep domain adaptation/ enables us to get closer/ to human-level performance/ in terms of the amount of training data. (원하는 task data (to be relative real-scene)가 적을 때 유용하게 사용할 수 있는 방법이다.)
- <img src='https://user-images.githubusercontent.com/46951365/104808662-11c6a080-582b-11eb-8c5e-44b1f0ee7a57.png' alt='drawing' width='350' style="zoom:150%;" />

---

---



# 2. Domain adaptation - Boston Universiry

- Domain Adaptation에 대한 설명을 그림으로 아주 재미있게 표현해놓은 좋은 자료.
- 하지만 아래 내용은 참고만 할 것. 논문을 찾아 읽어봐야 한다.
- Applications to different types of domain shift
    1. From dataset to dataset
    2. From simulated to real control
    3. From RGB to depth
    4. From 3D-CAD models to real images
- models adapted without labels (NO labels in target domain)
    - adversarial alignment
    - correlation alignment
- D = distributions/ xi, zj = Image/ yi = Label :   
    <img src='https://user-images.githubusercontent.com/46951365/104808053-0a9d9380-5827-11eb-95ed-70fbb22eafe0.png' alt='drawing' width='300' style="zoom: 200%;" />



**(1) From dataset to dataset, From RGB to depth**

- DRCN와 유사하지만 좀 더 복잡한 형태의 confusion-loss 사용하는 논문
    - [Simultaneous Deep Transfer Across Domains and Tasks - citation 889](https://arxiv.org/pdf/1510.02192.pdf)
    - <img src='https://user-images.githubusercontent.com/46951365/104808822-0fb11180-582c-11eb-9752-e57992a01b77.png' alt='drawing' width='500' style="zoom: 150%;" />  
    - domain classifier loss : Domain이 source인지 target인지 판단한다. 잘못 판단하면 Loss가 커지므로, 잘 판단될 수 있도록 학습 된다.
    - domain confusion loss : Domain이 source인지 target인지 잘못 판단하게 만든다. 
    - **한번은 classifier loss로, 다른 한번은 confusion loss로 학습**시키므로써, Network가 Source에서만 잘 동작하게 만드는게 아니라, Target에서도 잘 동작하게 만든다. 
    - **지금의 feature extractor는 너무 source중심으로 domain loss통해서 target에 대한 정보도 feature extractor가 학습하게 만드는 것이다.**
    - Target으로 Test해봤을 때, 그냥 Source만으로 학습된 Network를 사용하는 것보다 이 과정을 통해서 학습된 Network에서 더 좋은 Accuracy 결과가 나왔다.
- ADDA
    - [**Adversarial Discriminative Domain Adaptation (2017 - citation 1871)**](https://arxiv.org/abs/1702.05464)
    - 위의 Encoder 구조에서 Weight sharing을 하지 않음.
    - RGB-Depth 추측에서도 좋은 성능 획득



**(2) From simulated to real control** 

- [Transfer from Simulation to Real World through Learning Deep Inverse Dynamics Model](https://arxiv.org/abs/1610.03518)



**(3) From 3D-CAD models to real images**

- Domain Adaptation via **Correlation Alignment**  
- [Deep CORAL: Correlation Alignment for Deep Domain Adaptation](https://arxiv.org/pdf/1607.01719.pdf) 논문 참조.  

<img src='https://user-images.githubusercontent.com/46951365/104809729-9cf76480-5832-11eb-8c55-7ccf0bc2dcc6.png' alt='drawing' width='400' style="zoom:150%;" />




