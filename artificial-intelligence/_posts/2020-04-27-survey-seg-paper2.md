---
layout: post
title: (논문) Image Segmentation Using Deep Learning -A Survey [2]
description: >  
    Image Segmentation Using Deep Learning: A Survey 논문 리뷰 및 정리
---

(위성Segment) Segmentation Survey 논문 정리 2
논문 원본 : **2020 (cite:6) [Image Segmentation Using Deep Learning: A Survey](https://arxiv.org/abs/2001.05566)** 

# Section 3: DeepLearning -BASED IMAGE SEGMENTATION MODELS

- 여기부터 2019년 이후에 제안된 100가지가 넘는 segmentation methods를 상세히 검토한다.
- 많은 방법론에서 이것을 사용해서 좋은 성과를 얻었다. 
  - encoder and decoder parts
  - skip-connections 
  - multi-scale analysis 
  - the use of dilated convolution... more recently 
- 많은 methods를 특징(contributions)로 분류하기는 어렵고, Section2의 방법으로 분류하려한다.

## 3.1 Fully Convolutional Networks

1. semantic image segmentation을 위한 최초의 신경망이라고 할 수 있다. 
2. VGG16 및 GoogLeNet를 사용하였고, Fully connected layer와 같이 conv 연산이 아닌 layer를 모두 convolutional layers로 바꾸었다. (fig.7)
3. 임의 사이즈의 input을  받아도 적절한 크기의 segmentation map output을 도출하는 신경망이다. 
4. skip connections을 사용했다. 그러기 위해 upsampling을 적절히 사용하여,  [from deep, coarse layers] 정보와 [from shallow, fine layers]정보를 융합하였다. (fig. 8)
5. PASCAL VOC, NYUDv2, SIFT Flow Dataset에서 SOTA를 달성하였다.

<img src="https://user-images.githubusercontent.com/46951365/80369871-f2b56680-88c9-11ea-85c5-eb16e12ebe87.png" alt="image" style="zoom:50%;" />

6. FCN은 지금의 Image Segmentation에서 milestone으로 여겨지지만, 충분히 빠르지 않고, Global context information을 추출하기에 효과적인 모델이 아니다. 그리고 3D이미지로 transferable 하지 않다.(?)
7. 이러한 문제를 해결하기 위해 ParseNet이 등장했다. Global context information에 강하기 위해, he average feature for a layer를 사용하였다. 각 레이어의  특징 맵이 전체 이미지에 풀링되어, context vector가 생성된다. 이것은 정규화되고 풀링되어 초기 피쳐 맵과 동일한 크기의 새 피쳐 맵을 생성한다. 그런 다음이 특성 맵에 연결된다.(fig9. (c)) 
8. FCN은  뇌종양 segmentation [34], instance-aware semantic segmentation [35], 피부 병변 segmentation [36] 및 홍채 segmentation [37]와 같은 다양한 세그먼테이션 문제에 적용되어왔다. 

<img src="https://user-images.githubusercontent.com/46951365/80370008-2c866d00-88ca-11ea-9dbe-7aa7de84229e.png" alt="image" style="zoom: 80%;" />



## 3.2 Convolutional Models With Graphical Models

1. FCN은  potentially useful scene-level semantic context(잠제적으로 유용한 전체적 정보)를 무시한다. 이러한 문제를 해결하기 위해  CRF (Conditional Random Field) 및 MRF (Markov Random Field)와 같은 확률 적 그래픽 모델을 통합하는 몇가지 방법이 있다.
2. CNNs and fully connected CRFs을 융합한 chen은 classification을 위해 high level task를 수행하는 depp CNN의 불가변적 속성(?) 때문에 정확한 Segmentation이 이루어지지 않음을 확인하였다. (fig 10)
3. 이러한 deep CNN의 문제를 해결하기 위해, CNN의 마지막 layer결과값과  fully-connected CRF를 결합하였고, 더 높은 정확도(sementation의 경계를 더 정확히 갈랐다.)를 가졌다. 

<img src="https://user-images.githubusercontent.com/46951365/80373211-b08f2380-88cf-11ea-860d-faa25436d671.png" alt="image" style="zoom:67%;" />

4. A fully-connected deep structured network for image segmentation. [39],[40]에서는 CNNs and fully-connected CRFs를 end-to-end로 학습시키는 network를 구조화하였다. PASCAL VOC 2012 dataset에서 SOTA 달성했다. 
5. [41]에서 contextual deep CRFs를 이용한 segmentation 알고리즘을 제안했다. 여기서는 contextual 정보를 뽑아내기 위해, “patch-patch"컨텍스트를  (between image regions)와“patch배경”컨텍스트를 사용했다.
6. [42]에서 high-order relations과 mixture of label contexts를 포함하는, MRFs 정보를 사용하였다. 과거의 MRFs방법과는 달리 deterministic end-to-end learning이 가능한 Parsing Network라는 CNN모델을 제안하였다.



## 3.3  Encoder-Decoder Based Models

- 대부분의 DL-based segmentation works에서 encoder-decoder models를 사용했다. 

- 우리는 아래와 같이 2가지 범주로 분류해 모델들을 확인해볼 것이다. 

  (general segmentation VS Medical and Biomedical segmentation)

3.3.1  Encoder-Decoder Models for General Segmentation
{:.lead}

1. 아래의 그림처럼 Deconvolution(= transposed convolution)을 사용한 Segmentation방법의 첫논문이 [43]이와 같다. (fig11) 그림과 같이 encoder와 decoder가 존재하고, decoder에서 deconvolution and unpooling layers를 사용해서 픽셀단위의 레이블링이 이루어진다. PASCAL VOC 2012 dataset에서 (추가 데이터 없이) 좋은 정확도가 도출되었다.

<img src="https://user-images.githubusercontent.com/46951365/80465304-f7375900-8975-11ea-967d-2dba45fc4671.png" alt="image" style="zoom:67%;" />

2. [44] SegNet은 the 13 convolutional layers in the VGG16 network와 구조적으로 동일한 encoder를 사용하였다. (fig12) 가장 중요한 점은 업샘플링을 하며 featrue map을 키울때 encoder에서 max pooling을 했던 그 인덱스를 기억해서 이용하는 것이다. 그리고 SegNet은 다른 모델에 비해서 매개변수가 적다는 장점이 있다. 
3. [45] SegNet의 upgrade version으로  A Bayesian version of SegNet은 encoder-decoder network의 불확실성(upsampling등의 문제)을 해결하고자 노력했다.

<img src="https://user-images.githubusercontent.com/46951365/80466178-226e7800-8977-11ea-9c51-75ada3315fb9.png" alt="image" style="zoom:67%;" />

4. 최근 개발된 유명한 high-resolution network (HRNet)은 (이전의 고해상도 표현을 복구하려는 DeConvNet, SegNet, U-Net and V-Net과는 조금 다르게) 아래의 사진처럼 고해상, 저해상 이미지와의 정보교환을 이뤄가며 좋은 성능을 얻었다. 
5. 최근의 많은 Semantic segmentation 모델들은 contextual models, such as self-attention을 사용하면서, HRNet을 Backbone으로 많이 사용합니다. 
6. 지금까지 봤던 Network이 외에도, transposed convolutions, encoder - decoders를 이용하는 최근 모델에는 Stacked Deconvolutional Network (SDN) [46], Linknet [47], W-Net [48], and locality-sensitive deconvolution networks for RGB-D segmentation [49]와 같은 모델들이 있습니다.

![image](https://user-images.githubusercontent.com/46951365/80488016-b3098000-8998-11ea-8eed-56872ba6777d.png)



<br>

3.3.2  Encoder-Decoder Models for Medical and Biomedical Image Segmentation
{:.lead}

- medical/biomedical image segmentation을 하기 위한 몇가지 모델들을 공부해본다.
- FCN과 encoder-decoder를 기본으로 하는 U-Net [50], and V-Net [51]이 의료분야에서 유명하다.

![image](https://user-images.githubusercontent.com/46951365/80489728-3a57f300-899b-11ea-8a4f-5f7e0c42fe74.png)

1. 현미경 이미지를 분류하기 위해 U-Net이[50] 개발되었고, 3D medical image(MRI volume) segmentation을 위해서 V-Net[51]이 개발되었다. () (자세한 설명은 논문 참조)
2. 그 외에 Unet은 3D images를 위한 [52], nested Unet[53], **road extraction [54]**에 많이 사용되었다. 또한 흉부 CT 영상 Progressive Dense V-net (PDV-Net)



## **3.4**  **Multi-Scale and Pyramid Network Based Models**






