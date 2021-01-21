---
layout: post
title: 【Paper】Two-phase Pseudo Label based Domain Adaptation
---

**논문** : [Open Compound Domain Adaptation](https://liuziwei7.github.io/projects/CompoundDomain.html)

**분류** : paperswithcode - Unsupervised Domain Adaptation

**저자** : Ziwei Liu1∗, Zhongqi Miao2∗, Xingang Pan, Xiaohang Zhan

**읽는 배경** : 판 페이 박사과정 선배가 추천해준 논문이자, 우상현 랩장과 박광영 박사과정 선배님이 최근에 발표하신 'Discover, Hallucinate, and Adapt'의 기본이 되는 논문이다. 

읽으면서 생각할 포인트 : 굉장히 어려운 논문이니 모르는게 많이 나와도 일단 달려가자. 핵심 key word를 중심으로 정리하면서 기록해두자. the present and problems and issues, ours Solution 흐름으로 정리하고 기록해놓자.



## **느낀점**  

1. 



# 질문&답변

1. 





## **<u>다 읽은 후, 필수로 읽어야 겠다고 생각이 든 논문</u>**

1. 



# 0. Abstract

- the present 
  - A typical domain adaptation approach이란? 
  - Clear distinctions between source and target
- Ours
  - open compound domain adaptation (OCDA) 의 2가지 핵심기술
    1. **instance-specific curriculum domain adaptation strategy** : generalization across domains / in a data-driven <u>self-organizing fashion(???)</u> 
    2. **a memory module** : the model’s agility(예민함, 민첩한 적응?) towards novel domains / memory augmented features for handling open domains. (메모리가 더 다양한 feature 추출에 도움을 준다.)
  - 실험을 적용해본 challenges
    1. digit classification
    2. facial expression recognition
    3.  semantic segmentation
    4. reinforcement learning



# 1. Introduction

-  <u>the present & problem</u>
  - Supervised learning : 좋은 성능이지만 비현실적
  - domain adaptation : 1 source : 1target 에 대해서 clear distinction를 정의하려고 노력하지만 이것도 비현실적. 현실은 많은 요소들(비,바람,눈,날씨)에 의해서 다양한 domain의 데이터가 함께 존재하므로.
- <u>Ours - open compound domain adaptation</u>
  - more realistic
  - adapt labeled source model /to **unlabeled compound target**
    - 예를 들어 SVHN [33], MNIST [21], MNISTM [7], USPS [19], and SynNum [7] 는 모두 digits-recognition인데, 이들을 모두 다른 domain이라고 고려하는 것은 현실적이지 않다.
  - 우리는 compound target domains로써 그들을 고려하고, unseen data까지 test해볼 계획이다. 
    - 기존의 domain adaptation : rely on some **holistic** measure of instance difficulty.
    - 우리의 domain adaptation : rely on their **individual** gaps to the labeled source domain
  - <u>네트워크 학습 과정</u>
    1. classes in labeled source domain를 discriminate(classify) 하는 모델 학습
    2. (source와 많이 다르지 않은) easy target를 넣어서 domain invariance를 capture하게 한다.
    3. source와 easy target 모두에서 잘 동작하기 시작하면, hard target을 feed한다.
    4. 이런 방식으로 classification도 잘하고, 모든 domain에서 robust한 모델을 만든다.
  - <u>Technical insight</u>
    1. Tech1 : domain-specific feature representations을 가지는 Classification Network에 대해서, source와의 feature distance가 가까운 target은 Network 변화에 많은 기여를 하지 않는 것을 이용한다. 그래서 **distill(증류해 제거해나간다,) the domain-specific factors** (즉 domain에 robust한 Network 제작)
    2. Tech2 : **memory module**이 inference를 하는 동안 open-domain에서도 정확한 feature를 추출하도록 도와준다. the input-activated memory features(input에 따라 다르게 행동하는 memory feature) 



# 2. Relative work

![image-20210121162754802](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210121162754802.png)

1. Unsupervised Domain Adaptation :
   - **1 source - 1 target**
   - cannot deal with more complicated scenarios 
   - 참고 논문들 : latent distribution alignment [12], backpropagation [7], gradient reversal [8], adversarial discrimination [48], joint maximum mean discrepancy [30], cycle consistency [17] and maximum classifier discrepancy [42].
2. Latent & Multi-Target Domain Adaptation :
   -  clear **domain distinction&labels**(domain끼리의 차이가 분명함) / not real-world scenario
   - 참고 논문들 : latent [16, 51, 32] or multiple [11, 9, 54] or continuous [2, 13, 31, 50] target domains
3. Open/Partial Set Domain Adaptation :
   - target이 source에 없는 카테고리를 가지거나, subset of categories 를 가지거나. 
   - “openness” of domains = unseen domain data에 대해서 고려하기 시작
   - 참고 논문들 : open set [37, 43] and partial set [55, 3] domain adaptation.
4. Domain Generalized/Agnostic Learning :
   - Learn domain-invariant universal representations (domain이 변하더라도 같은 특징을 잘 추출하는 신경망 모델)
   - 참고 논문들 : Domain generalization [52, 23, 22] and domain agnostic learning [39, 5]
5. <u>바로 위의 논문들의 문제점과 our 해결</u>
   - 문제점 : they largely neglect the latent structures inside the target domains
   - 해결책 : Modelling **the latent structures** inside the compound target domain by learning <u>**domain-focused factors(???)**</u>



# 3. Our Approach to OCDA










































