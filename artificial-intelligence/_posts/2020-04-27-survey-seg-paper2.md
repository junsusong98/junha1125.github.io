---
3layout: post
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
2. VGG16 및 GoogLeNet를 사용하였고, Fully connected layer와 같이 linear 연산이 아닌 layer를 모두 convolutional layers로 바꾸었다. (fig.7)
3. 임의 사이즈의 input을  받아도 적절한 크기의 segmentation map output을 도출하는 신경망이다. 
4. skip connections을 사용했다. 그러기 위해 upsampling을 적절히 사용하여,  [from deep, coarse layers] 정보와 [from shallow, fine layers]정보를 융합하였다. (fig. 8)
5. PASCAL VOC, NYUDv2, SIFT Flow Dataset에서 SOTA를 달성하였다.

<img src="https://user-images.githubusercontent.com/46951365/80369871-f2b56680-88c9-11ea-85c5-eb16e12ebe87.png" alt="image" style="zoom:50%;" />

6. FCN은 지금의 Image Segmentation에서 milestone으로 여겨지지만, 충분히 빠르지 않고, Global context information을 추출하기에 효과적인 모델이 아니다. 그리고 3D이미지로 transferable 하지 않다.(?)
7. 이러한 문제를 해결하기 위해 ParseNet이 등장했다. Global context information에 강하기 위해, he average feature for a layer를 사용하였다. 각 레이어의  특징 맵이 전체 이미지에 풀링되어, context vector가 생성된다. 이것은 정규화하고 언풀링하여 초기 피쳐 맵과 동일한 크기의 새 피쳐 맵을 생성한다. 그런 다음이 특성 맵에 연결된다.(fig9. (c)) 
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





## **3.4**  Multi-Scale and Pyramid Network Based Models

- 주로 Object Detection에서 Multi-scale analysis을 위해서 가장 많이 사용되는 Pyramid Network (FPN) [56]을 중심으로 사용한 모델들을 살펴보자. 
- FPN은 low and high resolution features를 융합하기 위해서, bottom-up pathway, a top-down pathway and lateral connections방법을 사용합니다. FPN 저자는 Segmentation을 위해서 각 층(다른 해상도 크기의)의 predict단에서 (간단히) 2 layer의 MLP를 사용하였습니다.

<img src="https://user-images.githubusercontent.com/46951365/80493227-1945d100-89a0-11ea-8010-f9aedecef9ae.png" alt="image" style="zoom: 50%;" />

1. [57] PSPN 모델은 (fig17) the global context representation of a scene를 더 잘 이해하기 위해 개발되었다. 특별한 점으로는 ResNet + dilated Network을 사용해서 (b) Featrue Map을 뽑아낸다는 것이다. 그리고 a pyramid pooling module을 사용해서 다른 크기의 pattern(객체)들을 구분해간다. pooling 이후에 1*1 conv를 사용해서 channel을 감소시키고, 마지막 단에는 지금까지의 모든 정보를 concat한 후 conv를 거쳐  픽셀 단위 예측을 수행합니다.
2. [58]는 고해상도 maps(w,h가 큰)의 skip connection을 이용해서. 저해상도 maps으로 부터 재생성된 segmentation결과에서 경계가 모호한 문제를 해결합니다. 
3. 이 외에 Multi-scale analysis를 사용한 모델로 DM-Net (Dynamic Multi-scale Filters Network) [59], Context contrasted network and gated multi-scale aggregation (CCN) [60], Adaptive Pyramid Context Network (APC-Net) [61], Multi-scale context intertwining (MSCI) [62], and salient object segmentation [63]. 등이 있다.





## 3.5  R-CNN Based Models (for Instance Segmentation)

- R-CNN 계열을 이용해 instance segmentation 문제를 해결하는데 많이 사용되었다. 특히 Faster-RCNN은 RPN을 사용해 ROI를 추출한다. 그리고 RoiPool와 CNN을 통해서 객체 위치와 객체 클래스를 유추한다.

1. Mask R-CNN [65]은 Faster-Rcnn에서 2개의 출력 분기가 아닌, 3개의 출력 분기를 사용하여, 각 객체에 대해서 Instance Segmentation을 수행한다. COCO test set에서 좋은 결과를 도출해 낸다.

   <img src="https://user-images.githubusercontent.com/46951365/80497902-0cc47700-89a6-11ea-8f9c-3ee7cfeb6a65.png" alt="image" style="zoom:50%;" />

2. Path Aggregation Network (PANet) [66]은 Mask R-CNN과 FPN(The feature extractor)을 기반으로 한다. 위의 사진과 같이 (b), (c)를 사용하는게 특징이고, (e)에서 처럼 Roi를 FCN처리를 하여 the object Mask를 예측한다.

3. [67]에서는 instances를 구별하고, estimating masks, categorizing objects 하기 위한  multi-task network를 개발한 것이 특징이다. [68]에서는 a novel weight transfer function,  a new partially-supervised training paradigm을 사용해서 많은 class instance segmentation수행(label을 정의하지 않은 객체에도, 비지도학습을 이용해서(box의 class값을 사용해서) 새로운 label을 예측하는)을 가능케 한 것이 특징이다. 

4. [69]에서는 Faster-RCNN을 개선한 MaskLab을 개발하였다. Roi에 대해서, semantic and direction prediction를 수행하여, segmentation을 수행하는 것이 특징이다.

5. [70]에서는 Tensormask를 개발하였다. 이 모델은 dense (sliding window instance) object segmentation에서 좋은 결과를 도출하였다. 4D 상태에서 Prediction을 수행하였고, tensor view를 사용해 Mask-RCNN과 유사한 성능을 가지는 모델을 만들었다.

6. RCNN을 기반으로 개발된 다른 모델로써, R-FCN [71], DeepMask [72], SharpMask [73], PolarMask [74], and boundary-aware in-stance segmentation [75]와 같은 것들이 있다. 

7. 또한  Deep Watershed Transform [76], and Semantic Instance Segmentation via Deep Metric Learning [77]에서 처럼, grouping cues for bottom-up segmentation을 학습함으로써  instance segmentation에서의 문제를 해결하려고 노력했다는 것도 눈여겨 봐야한다.



## 3.6 Dilated Convolutional Models and DeepLab Family

- Dilated convolution(=atrous conv)는 the dilation rate(a spacing between the weights of the kernel w)를 사용한다. 이로써 추가적 비용없이 the receptive field를 키울 수 있었다. 따라서 real-time segmentation에서 많이 사용된다. 

  - Dilated Conv를 사용한 많은 모델들 : the DeepLab family [78], multi-scale context aggregation [79], dense upsampling convolution and hybrid dilatedconvolution (DUC-HDC) [80], densely connected Atrous Spatial Pyramid Pooling (DenseASPP) [81], and the efficient neural network (ENet) [82]

  <img src="https://user-images.githubusercontent.com/46951365/80595338-88362f00-8a5f-11ea-9aa7-d5b8cd9141e0.png" alt="image" style="zoom: 67%;" />



1. DeepLab2에서는 3가지 주요한 특징이 있다. 

   - max-pooling를 하여 이미지의 해상도(w,h 크기)를 낮추는 대신에 dilated conv를 적극적으로 사용하였다.
   -  Atrous Spatial Pyramid Pooling(ASPP)를 사용해서 multiple scales object를 더 잘 탐사하였다. 
   - 객체 경계를 더 잘 구별하기 위해서 deep CNNs and probabilistic graphical models을 사용하였다. 

2. DeepLab은 2012 PASCAL VOC challenge, the PASCAL-Context challenge, the Cityscapes challenge에서 좋은 성능을 내었다. (fig.25)

3. 이후 Deeplabv3[12]에서는 cascaded and parallel modules of dilated convolutions(ASPP{1*1conv를 사용하고 배치 정규화를 사용하는}에서 그룹화된)를 사용하였다.  

4. 2018년 이후에 [83]에서 encoder-decoder architecture를 사용한 Deeplabv3+가 새로 발표되었다. 아래 2개의 기술을 사용한 것이 특징이다. 

   - a depthwise convolution (spatial convolution for each channel of the input)로 만들어진, atrous separable convolution
   -  pointwise convolution (1*1 convolution with the depthwise convolution as input)

   Deeplabv3+는 DeepLabv3를 encoder 프레임워크(backbone)로 사용하였다. (fig.26)

5. 최근에 수정된 Deeplabv3+ 모델에는 max-pooling와 batch Norm을 사용하지 않고, 더 많은 layers와  dilated depthwise separable convolutions를 사용하는 'Xception backbone'를 사용한다. 
6. Deeplabv3+는 the COCO and the JFT datasets을 통해 pretrained 된 모델을 사용하여, the 2012 PASCAL VOC challenge Dataset에서 놓은 성적을 얻었다.





![image](https://user-images.githubusercontent.com/46951365/80595563-e82cd580-8a5f-11ea-88fc-1be4559ed5c8.png)

<img src="https://user-images.githubusercontent.com/46951365/80597489-df89ce80-8a62-11ea-823b-7efb096ed134.png" alt="image" style="zoom: 67%;" />

<br>

## 3.7 Recurrent Neural Network Based Models

- segmentation을 수행할 때, 픽셀간의  the short/long term dependencies를 모델링할 때 RNN을 사용하는 것도 유용하다. RNN을 사용함으로써 픽셀들이 연결되어 연속적으로 처리된다.  그러므로써 global contexts를 좀 더 잘 찾아내고, semantic segmentation에서의 좋은 성능이 나오게 해준다. RNN을 사용함에 있어 가장 큰 문제점은 이미지가 2D 구조라는 것이다. (RNN은 문자열같은 1차원을 다루는데 특화되어 있으므로.)

1. [84]에서 Object detection을 위한 ReNet[85]를 기반으로 만들어진 ReSeg라는 모델을 소개한다. 여기서 ReNet은 4개의 RNN으로 구성된다. (상 하 좌 우) 이로써 global information를 유용하게 뽑아낸다. (fig.27)

![image](https://user-images.githubusercontent.com/46951365/80601928-6e99e500-8a69-11ea-875a-07477be3dda6.png)



2. [86]에서는 (픽셀 레이블의 복잡한 공간 의존성을 고려하여) 2D LSTTM을 사용해서 Segmentaion을 수행했다. 이 모델에서는 classification, segmentation, and context integration 모든 것을 LTTM을 이용한다.(fig.29)
3. [87]에서는 Graphic LSTM을 이용한 Segmentation모델을 제안한다. 의미적으로 일관된 픽셀 Node를 연결하고, 무 방향(적응적으로 랜덤하게) 그래프를 만든다. (edges로 구분된 그래프가 생성된다.)  fig.30을 통해서 Graph LSTM과 basic LSTM을 시각작으로 비교해볼 수 있다. 그리고 Fig31을 통해서 [87]의 전체적 그림을 확인할 수 있다. (자세한 내용은 생략)

![image](https://user-images.githubusercontent.com/46951365/80603862-ecf78680-8a6b-11ea-970f-b3dc93f3fcbd.png)

4. DA-RNNs[88]에서는 Depth카메라를 이용해서 3D sementic segmentation을 하는 모델이다. 이 모델에서는 RGB 비디오를 이용한 RNN 신경망을 사용한다. 출력에서는 mapping techniques such as Kinect-Fusion을 사용해서 semantic information into the reconstructed 3D scene를 얻어 낸다. 

<img src="https://user-images.githubusercontent.com/46951365/80604136-4c559680-8a6c-11ea-8e19-2d5edffc4cca.png" alt="image" style="zoom: 80%;" />

5. [89]에서는 CNN&LSTM  encode를 사용해서 자연어와 Segmentation과의 융합을 수행했다. 예를 들어 "right woman"이라는 문구와 이미지를 넣으면, 그 문구에 해당하는 영역을 Segmentation해준다.  visual and linguistic information를 함께 처리하는 방법을 학습하는 아래와 같은 모델을 구축했다. (fig.33) 
6. LSTM에서는 언어 백터를 encode하는데 사용하고, FCN을 사용해서 이미지의 특성맵을 추출하였다. 그렇게 최종으로는 목표 객체(언어백터에서 말하는)에 대한 Spatial map (fig 34의 가장 오른쪽)을 얻어낸다. 

<img src="https://user-images.githubusercontent.com/46951365/80604535-ce45bf80-8a6c-11ea-9f19-2682077cf1d8.png" alt="image" style="zoom:80%;" />



<br>

## 3.8 Attention-Based Models

- [Attention 메카니즘 기본 이론]([https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/10/06/attention/](https://ratsgo.github.io/from frequency to semantics/2017/10/06/attention/))
- 사실 Attention 알고리즘은 컴퓨터 비전에서 지속적으로 연구되어 왔었다.

1. [90]에서는 multi-scale feature를 사용해서 attention해야할(wdight) 정도를 학습한다. pooling을 적용하는 것보다, attention mechaniism을 사용함으로써 다른 위치 다른 scales(객체크기)에 대한 the importance of features(그 부분을 얼마나 중요하게 집중할지)를 정확하게 판단할 수 있다. (fig. 35)



<img src="https://user-images.githubusercontent.com/46951365/80677600-24aa1100-8af4-11ea-81ff-eb926d46c4ad.png" alt="image" style="zoom: 67%;" />

<img src="https://user-images.githubusercontent.com/46951365/80677585-1c51d600-8af4-11ea-96e6-25ed433ad3d4.png" alt="image" style="zoom:67%;" />

2. [91]에서는 일반적인 classifier처럼 labeled된 객체의 의미있는 featrue를 학습하는게 아니라,  reverse attention mechanisms(RAN)을 사용한 semantic segmentation방법을 제안하였다. 여기서 RAN은 labeled된 객체만을 학습하는 것이 아닌, 반대의 것들(배경, 객체가 아닌 것)에 대한 개념도 캡처(집중)한다. (fig. 36)
3. [92]에서는  a Pyramid Attention Network를 사용했다. 따라서 global contextual information을 사용하는데 특화되어 있습니다. 특히 이 모델에서는 dilated convolutions and artificially designed decoder networks를 사용하지 않습니다. 
4. [93]에서는 최근에 개발된 a dual attention network을 사용한 모델을 제안한다.  중심 용어 - rich con-textual dependencies / self-attention mechanism /  the semantic inter-dependencies in spatial  +  channel dimensions / a weighted sum 
5. 이 외에도 [94] OCNet(object context pooling), EMANet [95], Criss-Cross Attention Network (CCNet) [96], end-to-end instance segmentation with recurrent attention [97], a point-wise spatial attention network for scene parsing [98], and a discriminative feature network (DFN) [99] 등이 존재한다.



<br>

## 3.9 Generative Models and Adversarial Training

<img src="https://user-images.githubusercontent.com/46951365/80678950-dfd3a980-8af6-11ea-9277-d98067f27c91.png" alt="image"  />

1. [100]에서는 segmentation을 위해서, adversarial training approach를 제안하였다. (fig. 38)에서 보는 것처럼 Segmentor(지금까지 했던 아무 모델이나 가능) segmentation을 수행하고, Adversarial Network에서 ground-truth와 함께, discriminate를 수행해나간다. 이러한 방법으로 Stanford Background dataset과 PASCAL VOC 2012에서 더 높은 정확성을 갖도록 해준다는 것을 확인했다. (fig. 39)에서 결과 확인 가능하다.
2. [101]에서는 semi-weakly supervised semantic segmentation using GAN를 제안하였다.
3. [102]에서는 adversarial network를 사용한 semi-supervised semantic segmentation를 제안하였다. 그들은 FCN discriminator를 사용해서 adversarial network를 구성하였고, 3가지 loss function을 가진다. 1. the segmentation ground truth와 예측한 결과와의 cross-entropy loss 2. discriminator network 3. emi-supervised loss based on the confidence map. (fig. 40)
4. [103]에서는 의료 영상 분할을 위한 multi-scale L1 Loss를 제안하였다. 그들은 FCN을 사용해서 segmentor를 구성했고, 새로운 adversarial critic network(multi-scale Loss)를 구상했다. critic(discriminator)가 both global and local features에 대한 Loss를 모두 뽑아낸다. (fig.41) 참조. 
5. 다른 segmentation models based on adversarial training로는, Cell Image Segmenta-tion Using GANs [104], and segmentation and generation of the invisible parts of objects [105]이 있다.



<br>

## 3.10 CNN Models With Active Contour Models

- Active Contour Models (ACMs) [7] 를 기본으로 사용하여, Loss function을 바꾼 새로운 모델들이 새롭게 나오고 있다. 
  - the global energy formulation of [106]
  -  [107] : MRI 영상 분석
  - [108] : 미세 혈관 이미지 분석
- [110] ~ [115]에 대한 간략한 설명은 논문 참조.



## 3.11  Other Models + Popular Models Timeline

위의 모델을 이외의, DL architectures for segmentation의 몇가지 유명한 모델들을 간략히 소개한다.

- [116] ~ [140]
- fig. 42는  semantic segmentation를 위한 아주 유명한 모델들을 시간 순으로 설명한다. 지난 몇 년간 개발 된 많은 작품을 고려했을때, 아래의 그림은 가장 대표적인 작품 중에서도 일부만을 보여준다.

![image](https://user-images.githubusercontent.com/46951365/80683802-b4a18800-8aff-11ea-949e-e480e153b16d.png)













































