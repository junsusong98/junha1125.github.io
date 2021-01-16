---
layout: post
title: 【Domain-Adaptation】Deep Domain Adaptation Basics
---
Deep Domain Adaptation Basics

# Reference (Study order)
1. [https://towardsdatascience.com/deep-domain-adaptation-in-computer-vision-8da398d3167f](https://towardsdatascience.com/deep-domain-adaptation-in-computer-vision-8da398d3167f)
2. [http://www.eecs.umich.edu/eecs/pdfs/events/4142.pdf](http://www.eecs.umich.edu/eecs/pdfs/events/4142.pdf)
3. [https://sudonull.com/post/9686-Overview-of-Deep-Domain-Adaptation-Basic-Methods-Part-1](https://sudonull.com/post/9686-Overview-of-Deep-Domain-Adaptation-Basic-Methods-Part-1)

# 1. Deep Domain Adaptation In Computer Vision
## 1. introduction
  - NYC dataset로 학습시킴. Manhattan에서는 잘 동작. Paris에서 사용하니 문제 발생! 
  - images from different viewing angles / different lighting conditions 
    - source distribution, source dataset : Pre-train을 위해서 사용했던 데이터셋(의 분포, 분야)
    - targer distribution, source dataset : domain changed 된, dataset.
  - 일반적으로, damain adaptation은 한개 혹은 그 이상의 데이터셋을 사용한다. 
  - Approaches
    - shallow (not deep) : reweighting the source samples. trying to learn a shared space to match the distributions. (데이터 분포의 공통 부분을 학습하기 위해, source samples에 대해 다시 학습)
    - Deep : **아래 layer에서 배우는 학습값은 transferable representations** (inference에 많은 영향을 끼킬 수 있는. 결과가 크게 변할 수 있는.) 이다. **반대로 윗 layer에서 배우는 값은 sharply transferability가 떨어진다.** 
    - Deep Domain Adaptation 에서 이러한 성질을 사용한다. 
## 2. Domain Adaptation Categories
  - 참고 논문 
    1. Deep Visual Domain Adaptation: A Survey (2018)
    2. A Survey of Unsupervised Deep Domain Adaptation (2019)
    3. 과제의 복잡성, labeled/unlabeld data의 양, source/target의 차이 정도.로 분류
  - domain adaptation : the task space is the same(해야하는 과제는 같다. maybe Ex. obejct detection, segment. , kinds of labels) But input domain is divergent.
  - 크게 분류
    - homogeneous (동족의) : input feature spaces 는 같지만, data distributions이 다르다. 
    - heterogeneous (여러 종류,종족의) : feature spaces and their dimensionalities(차원수) 다르다.
    - 혹은
    - one-step domain adaptation (common type)
    - multi-step domain adaptation (요즘 사용 X)
  - target data에 따라서 
    - supervised : target data is labeled
    - semi-supervised : labeled and unlabeled
    - self-supervised : No labeled data
## 3. Task Relatedness
  - task relatedness : source가 task에 얼마나 적응될 수 있는지
  - Task Relatedness를 정의(수치화)하는 방법
    1. how close their **parameter vectors**. 
    2. how **same features** 
    3. 각각의 데이터가 정해진(fixed) 확률분포에 의해서 생성된 데이터라면, 서로 (transformations) F-related (Task Relatedness가 크다) 하다.
  - 그래도 domain adaptation을 사용할 수 있는지는, 학습을 시켜보고 이점을 얻을 수 있는지 없는지를 직접 확인해보는 방법 뿐이다. 
## 4. One-Step Domain Adaptation
3 basic techniques
1. divergence-based domain adapatation.
2. adversarial-based domain adaptation using GAN, domain-confusion loss.
3. reconstruction using stacked autoencoders.

하나씩 알아가보자.  
1. Divergence-based Domain Adaptation
    - **some divergence criterion (핵심 차이점 기준?) 을 최소화**하고, **domain-invariant feature representation** 을 찾아내는 것(achieve) (domain 변화에도 변하지 않는 feature extractor Ex. 어떤 신호등 모양이든 같은 신호등feature가 나오도록)
    - (1) Maximum Mean Discrepancy (MMD), (2) Correlation Alignment (CORAL), (3) Contrastive Domain Discrepancy (CCD) 
    - (1) MMD 
      - RKHS(Reproducing Kernel Hilbert Space)에 두 데이터를 mapping 후, 두 데이터의 feature 의 **평균을 비교**함으로써 두 데이터가 같은 분포의 데이터인지 확인한다.
      - 평균이 다르면, 다른 분포 데이터. 
      - 두 분포가 동일한 경우, 각각 분포의 샘플 간의 평균 유사성은 두 분포를 합한 샘플 간의 평균 유사성과 동일할 것.(?) 이라는 직관을 이용한다.
      - <img src='https://user-images.githubusercontent.com/46951365/104798356-85f04c80-5809-11eb-8d0d-4fae09915519.png' alt='drawing' width='350'/>  
      - two-stream architecture는 파라메터 공유하지 않음. 
      - Soft-max, Regularization(Domain-discrepancy) loss를 사용해서, two-architecture가 **similar feature representation(=extractor)**가 되도록 만든다.
    - (2) CORAL : 
      - <img src='https://user-images.githubusercontent.com/46951365/104798534-f2b81680-580a-11eb-9507-7881188b601d.png' alt='drawing' width='400'/>
      - (b)처럼 distribution만 맞춘다고 해서 해결되지 못한다. (c)처럼 soure에 target correlation값을 추가함으로써 align시켜준다.
      - align the second-order statistics (correlations) instead of the means
      - [좋은 논문](https://arxiv.org/pdf/1607.01719.pdf) : Using a differentiable CORAL loss.
    - (3) CCD : 
      - label distributions 을 사용한다. (라벨별 확률 분포) by looking at conditional distributions(조건적인 P(확률분포\|특정라벨))
      - 두 데이터의 각 라벨에 대해, 교집합 domain feature를 찾는다. 
      - minimizes the intra-class discrepancy, maximizing the inter-class discrepancy.
      - [좋은논문](https://arxiv.org/pdf/1901.00976.pdf) : target labels are found by **clustering**. CCD is minimized.
    - 이런 방법으로
      1. 이런 방법을 optimal transport 라고 한다. 
      2. 두 데이터 간의 feature and label distributions 가 서로 비슷해 질 것이다.
      3. **두 architecture(extractor, representation)** 간의 차이가 줄어들게 만든다.
2. Adversarial-based Domain Adaptation
    - 






# 2. Domain adaptation - Boston Universiry
