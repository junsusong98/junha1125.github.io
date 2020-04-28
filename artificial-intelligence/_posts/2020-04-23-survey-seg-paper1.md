---
layout: post
title: (논문) Image Segmentation Using Deep Learning -A Survey [1]
description: >  
    Image Segmentation Using Deep Learning: A Survey 논문 리뷰 및 정리
---
(위성Segment) Segmentation Survey 논문 정리 1  
논문 원본 : **2020 (cite:6) [Image Segmentation Using Deep Learning: A Survey](https://arxiv.org/abs/2001.05566)** 



추가로 공부해야 하는 내용 

{:.lead}

1. Autoencoder
2. LSTM

# Abstract

1. Image segmentation은 이미지 처리 및 컴퓨터 비전의 핵심이다. 
2. 이 논문에서는 Segmentation을 위해, 선구적인 기술들에 대한 Survey를 다룬다. 
   - 예를 들어, 1. fully convolutional pixel-labeling networks, 2. encoder-decoder architectures, 3. multi-scale and pyramid based approaches, 4. recurrent networks, 5. visual attention models, and 6. generative models in adversarial settings.
3. 각 모델에 대해서, 널리 사용되는 데이터셋을 사용해서, <u>유사성, 강점(장점), 성능</u>을 비교한다. 
4. 마지막으로 독자들에게 <u>유망한 미래 연구&공부 방향도 제시</u>해준다. 

 

# Instruction

1. image segmentation의 필요성(중요성): 

   image segmentation은 시각적 이해 시스템(visual understanding system)에 있어 필수적인 요소이다.

   의료 영상 분석 (예 : 종양 경계 추출 및 조직 체적 측정), 자율 주행 차량 (예 : 탐색 가능한 표면 및 보행자 감지), 비디오 감시 및 증강 현실을 포함하여 광범위한 응용 사례에 있어 중요한 역할을 수행한다.

2. image segmentation 연구에 활용된 알고리즘:

   과거 :

   - thresholding
   - histogram-based bundling
   - region-growing
   - k-means clustering 
   - watersheds

   더 나아가:

   - active contours
   - graph cuts
   - conditional and Markov random fields
   - sparsity-based  methods

3. 최근 경향:

   Deep Learning을 이용한 image segmentation 모델들이 많이 나왔다. 

4. image segmentation의 종류?

   image segmentation은 semantic label에 대한 각 픽셀의 분류 문제(semantic segmentation) 혹은 개별 객체의 분할(instance segmentation)의 문제로 공식화 할 수 있습니다.

   - semantic segmentation는 모든 이미지 픽셀에 대해 일련의 객체 범주 (예 : 사람, 자동차, 나무, 하늘)를 사용하여 픽셀 수준 레이블링을 수행하므로 일반적으로 전체 이미지에 대한 하나의 label을 예측(predict)하는 이미지 분류보다 어려운 작업입니다.
   - instance segmentation은 이미지에서 관심있는 각 객체를 감지하고 묘사함으로써 semantic segmentation 범위를 더 확장합니다 (예 : 개인의 partitioning). 

5. 주요 기술 Contributions(특성, 계보)를 기준으로, 100개의 모델들을 다음 범주로 그룹화 할 수 있다. 

   1)   Fully convolutional networks

   2)   Convolutional models with graphical models

   3)   Encoder-decoder based models

   4)   Multi-scale ad pyramid network based models

   5)   R-CNN based models (for instance segmentation)

   6)   Dilated convolutional models and DeepLab family

   7)   Recurrent neural network based models

   8)   Attention-based models

   9)   Generative models and adversarial training

   10)   Convolutional models with active contour models

   11)   Other models

6. 이 논문을 전체요약하면 다음과 같이 적을 수 있다.

   1)  2019년 이후의 100개 이상의 Segmentation 알고리즘을 10가지 범주로 그룹화 했다.

   2)  학습 데이터, 네트워크 아키텍처, Loss 함수, 학습 전략, 모델의 주요 특성(어떤 계보를 따르는가?)를 달리하여 만든 많은 모델들의 Review, 통찰력 있는 분석 결과를 제시한다. 

   3) 2D, 3D를 포함한 20개의 인기 있는 segmentation 데이터 셋의 개요를 제공 한다.

   4) 이 모델들을 비교, 요약하여 앞으로의 Image segmentation의 몇가지 과제와 해결방향 그리고 잠재적 미래 방향을 제시한다.

7. 이 논문을 각 파트를 전체요약하면 다음과 같이 적을 수 있다.

   1) Section 2 : 모든 모델의 Backbone이 되는 인기 있는 deep neural network architectures 개요 제공

   2) Section 3 : 100개 이상의 중요한 Segmentation 모델에 대한 포괄적 개요, 그리고 그들의 장점, 특성 제공

   3) Section 4 : 유명한 Segmentation Dataset의 개요와 각각의 특성을 검토

   4) Section 5.1 : Segmentation 성능 지표 검토

   5) Section 5.2 : 모든 모델들의 (위에서 설명한 성능 지표에 따른) 정량적 성능 검토

   6) Section 6 : Image sementation의 주요 과제와 해결방향, 향후 방향들을 검토

   7) Section 7 : 결론 



# Section 2: Overview of Deep Neural Networks

## 2.1 Convolutional Neural Network (CNNs)

1. CNN은 주로 3 가지 유형의 layer로 구성된다.

   1) convolutional layer: 특징을 추출하기 위해 가중치 (kernel) (또는 필터)의 가중치 (kernel)가 있다.

   ii) nonlinear layer: 네트워크에 의한 비선형 함수의 모델링을 가능하게하기 위해 피쳐 맵에 활성화 함수를 적용한다. (보통 요소별로)

   iii) pooling layer: 특징 맵의 작은 이웃을 이웃에 대한 일부 통계 정보 (평균, 최대 등)로 대체하고 공간 해상도를 감소시킨다.

4. 레이어의 단위는 로컬로 연결되어 있다. 

   즉, 각 유닛은 이전 계층의 유닛의 수용 장으로 알려진 작은 이웃으로부터 가중 된 입력을 수신한다.

   다중 해상도 피라미드를 형성하기 위해 레이어를 쌓으면서 더 높은 레벨의 layer는 점점 더 넓은 수용 영역에서 기능을 학습합니다. 

5. CNN의 주요 계산 이점: 

   레이어의 모든 수용 필드가 가중치를 공유하여 완전히 연결된 신경망보다 훨씬 적은 수의 매개 변수를 생성한다는 것이다.

4. CNN 아키텍처:

   AlexNet // VGGNet // ResNet // GoogLeNet // MobileNet // DenseNet

## 2.2 Recurrent Neural Networks (RNNs) and the LSTM

1. RNN은 음성, 텍스트, 비디오 및 시계열과 같은 순차적 데이터를 처리하는 데 널리 사용된다.

   여기서 주어진 시간 / 위치의 데이터는 이전에 발생한 데이터에 의존한다. 

2. RNN의 구조

   ![image](https://user-images.githubusercontent.com/61573968/80074444-1f911300-8584-11ea-8afd-28c7c0ec734e.png)

   각 타임 스탬프에서 모델은 현재 시간 Xi의 입력과 이전 단계 hi-1의 숨겨진 상태를 수집하고 목표 값과 새로운 숨겨진 상태를 출력한다.

3. RNN의 한계점?

   RNN은 일반적으로 많은 실제 응용 프로그램에서 장기적인 종속성(long-term dependencies)을 캡처 할 수 없다. 이 때문에 gradient vanishing 또는 exploding 문제로 고통받는 경우가 많으므로 일반적으로 long sequences에 있어 한계점이 있다.

4. 위와 같은 RNN의 한계점을 극복하기 위해 LSTM(Long Short Term Memory)이 등장하였다.

5. LSTM 구조

   ![image](https://user-images.githubusercontent.com/61573968/80075314-79dea380-8585-11ea-8b2e-6362297f90de.png)

   LSTM 아키텍처에는 메모리 셀로 들어오고 나가는 정보의 흐름을 조절하는 3 개의 게이트 (input gate, output gate, forget gate)가 포함되어 임의의 시간 간격에 걸쳐 값을 저장한다.

6. input, forget states 및 다른 gate 사이의 관계는 다음과 같다.

   ![image](https://user-images.githubusercontent.com/61573968/80075602-eb1e5680-8585-11ea-9ad2-2e074bf39501.png)

   - $$x_t \Subset R^d$$ : time-step $$t$$의 input
   - $$d$$ : 각 word의 feature dimension
   - $$\sigma$$ : 요소별 시그모이드 함수 ([0,1])
   -   $$\bigodot$$: 요소별 곱셈
   - $$c_t$$ : 메모리 셀 ( gradient vanishing/exploding의 위험을 낮추기 위함, 이를 통해 따라서 기존 RNN에서 실행 가능한 오랜 기간 동안 종속성 학습 가능)
   - $$f_t$$ : forget gate (메모리셀을 reset 하기 위한)
   - $$i_t$$ : input gate (메모리 셀의 input을 제어)
   - $$o_t$$ : output gate (메모리 셀의 output을 제어)

## 2.3 Encoder-Decoder and Auto-Encoder Models

<img src="https://user-images.githubusercontent.com/46951365/80072708-9547af80-8581-11ea-8c2e-3d3348486759.png" alt="image" style="zoom: 33%;" />

1. 인코딩 함수는 입력을 잠재 공간 표현(Latent Representation)으로 압축한다. 
2. 디코더 함수는 위에서 만든 Latent Representation을 이용해서 출력을 만들어낸다(예측한다).
3. 여기서 Latent Representation는 입력의 주요한 feature들을 표현한 것이다 
4. Loss는 reconstruction loss라고 불리우는 L(y; y^)를 사용한다.
5. NLP에서 많이 사용된다. Output이 초해상화된 사진, Segmentation 결과 등이 될 수 있다.

**Auto-Encoder Models**  
{:.lead}  

1. Input과 output이 동일한 특별한 경우에 사용한다. (즉 input과 같은 output 생성 원할 때) 

2. 2가지 종류가 있다. 

   (1) SDAE (stacked denoising auto-encoder)

   - 가장 인기 있음. 여러개의 auto-encoder를 쌓아서 이미지 deNoising 목적에 많이 사용

   (2) VAE (variational auto-encoder)

   - Latent representation에 prior distribution(확률 분포)의 개념을 추가 시킨다.
   - 위의 확률 분포를 사용해서 새로운 이미지 y를 생성시킨다.

   ps. (3)  adversarial auto-encoders

   - prior distribution와 유사한 latent representation를 만들기 위해 adversarial loss를 사용

## 2.4 GANs(Generative Adversarial Networks)

1. 2개의 network 모델 : Generator(위조 지폐 제작사)  VS  Discriminator(경찰. 감별사)
2. x : 실제 이미지/ z : 노이즈 이미지/ y : G가 생성한 이미지

<img src="https://user-images.githubusercontent.com/46951365/80075518-c32ef300-8585-11ea-8b8d-5c97f765490c.png" alt="image" style="zoom: 67%;" />

3. 초기 GAN 이후의 발전
   - (Mr. Radford) fully-connected network가 아니라, convolutional GAN model 사용
   - (Mr. Mirza) 특정 label 이미지를 생성할 수 있도록 class labes로 conditional된 GAN 
   - (Mr. Arjovsky) 새로운 loss function 사용. y x의 확률 분포가 완전히 겹치지 않게 한다. 
   - 추가 : https://github.com/hindupuravinash/the-gan-zoo

## 2.5 Transfer Learning

1. 이미 학습된 모델은 이미지에서 the semantic information를 뽑아낼 수 있는 능력을 가지므로,  많은 DL(deep learning)모델들은 충분한 train data가 없을 수 있다. 이때 Transfer Learning이 효율적이다.

2. 다른 곳에 사용되던 model을 repurpose화(우리의 데이터에 맞는 신경망으로 학습)시키는 것이다. 

3. Image segmentation에서 많은 사람들은 (인코더 부분에서) ImageNet(많은 데이터셋)으로 학습된 모델을 사용한다. 

   































