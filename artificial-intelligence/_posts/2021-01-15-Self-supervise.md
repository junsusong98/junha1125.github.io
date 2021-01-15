---
layout: post
title: 【Self-Supervise】 Self-Supervised-Learning Basics 
---
Self-Supervised-Learning Basics 

# Self-Supervised-Learning Basics on webs
- Supervise 즉 데이터의 사람이 만든 'annotation Label'이 없어도, **Feature Extractor(Pre-trained Model)**을 만들 수 있는 몇가지 방법들에 대해 기초만 알아보자.

# 1. reference

- [https://project.inria.fr/paiss/files/2018/07/zisserman-self-supervised.pdf](https://project.inria.fr/paiss/files/2018/07/zisserman-self-supervised.pdf)
- [https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html)
- [https://www.fast.ai/2020/01/13/self_supervised/](https://www.fast.ai/2020/01/13/self_supervised/)


# 2. \<zisserman-self-supervised\> - 2018.07

1. 모델에 대한 상세한 설명은 없다. 하지만 전체적은 윤각선을 만들기 위해 먼저 빠르게 보기로 했다. 
2. Supervise-learning : With a large-scale dataset labelled / Can Specify a training loss / Easy to Train the network
2. why self-supervision : high Expense of dataset / supervision-starved / vast numbers of unlabelled images/videos
3. how infants may learn : The Development of Embodied Cognition
4. What is Self-Supervision?

   - **supervision**에게 데이터/pretrain결과를 제공하기 위한 A form of unsupervised learning.
   - withhold some part of the data (일부 데이터를 주지 않고) and task the network with predicting it. (예측하라고 과제 준다.)
   - with a proxy loss (대체된 손실값),  the Network learn what we really care about.
5. Three parts
   - From Images
   - From videos
   - From videos with sound

## 2.Part (1) - Images

   - (2014) relative positioning : Origin Image만으로도 학습이 가능하다. 이걸로 Pretrained를 시키고 나중에 supervise로 refine 학습을 진행한다. 
   - (2014) colourization 
   - (2014) exemplar networks : Perturb/distort image crops, Train to classify as same class
   - 학습 절차
     - backbone에 self-supervised training을 먼저 수행한다. using 위의 다양한 방법들.
     - backbone freeze!
     - 그리고 train classifier.
     - 아래와 같이 supervised를 사용한것 보다는 낫지만, 그래도 성능이 좋아지고 있다. 
     - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210115124531206.png?raw=tru" alt="image-20210115124531206" style="zoom:50%;" />
   - (2018) Image Transformation : Unsupervised representation learning by predicting image rotations
     - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210115134434239.png?raw=tru" alt="image-20210115134434239" style="zoom:50%;" />
     - Rotaion로 학습한 AlexNet에서 좋은 성능이 나왔다. then Relative Position, Colourization.

## 2.Part (2) - Videos

- What can we use to define a proxy loss 
  - Temporal order of the frames / Nearby (in time) frames / Motion of objects

- ### Video sequence order 

  - (frames order 1 -> 2 -> 3 vs 2 -> 1 -> 3 )
  - **Shuffle and Learn**
    - Using 'high motion window'(동영상 중에서도 핵심적인 Action 부분 3장) of Dataset (Action Reognition) that is imformative training tutples
    - Input : Tuple Image -> Return : correct order Tuple
    - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210115135553374.png?raw=tru" alt="image-20210115135553374" style="zoom:50%;" />
    - 위와 같이 공유되는 네트워크를, Sequence order를 이용해서 학습시키고, 그 네트워크를 사용해서 'Action Label 예측 모델'/'Human Pose Estimation' 모델을 fine-tuning한다. 
  - **Odd-One-Out Networks**
    - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210115135851163.png?raw=tru" alt="image-20210115135851163" style="zoom: 67%;" />
    - 위 방법으로 학습시킨, pre-train 모델을 사용해서 Shuffle and Learn 보다 좋은 성능 획득
  - Summary
    - **Important to select informative data in training** : 중요한 정보를 선택해서 그것으로 학습을 시켜야 한다. 예를 들어 회전, 동영상 순서 등 data에서 하나의 중요한 정보라고 할 수 있다.
    - **shuffle and learn을 통해서, 우리가 모르게 모델은 'Human Pose Estimation' 문제에서 필요한 feature extractor를 만들고 있었을 것이다.**

- ### Video direction 

  - (video playing forwards or backwards, 정방향 재생중인가? 뒤로 거꾸로 재생중인가? 1 -> 2 -> 3 vs 3 -> 2 -> 1 )) 
  - gravity, entropy, friction, causality 와 같은 물리적 법칙들에 대한 feature extractor를 생성할 것이다.
  - **T-CAM Model**
    - Input: optical flow in two chunks
    - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210115141419097.png?raw=tru" alt="image-20210115141419097" style="zoom: 67%;" />
    - 중요한 장면 흐름의 동영상만 이용해 학습을 하고.. 
    - 위의 pre-train network를 이용해서, Fine tune & test network on UCF101 (human action classification). so Network get the best performance ever!

- ### Video tracking 

  - Tracking Emerges by Colorizing Videos (2018)
    - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210115142442479.png?raw=tru" alt="image-20210115142442479" style="zoom:50%;" />
    - color 와 object tracking에는 분명한 관계가 있다!! 위의 그림처럼  같은 색을 tracking 하면 그게 적절한 object tracking이 될테니까!!
    - Input : reference Frame (Reference Color 존재), Input Frame(monochrome 흑백)
    - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210115143259887.png?raw=tru" alt="image-20210115143259887" style="zoom: 67%;" />
    - 여기서 A는 공간유사도(내적은 두 백터가 유사할 수록 큰 값이 나오므로.) C_i는 참고하는 부분의 color.  즉 **∑Ac**는 참고 공간을 전체 훓어 봤을때, 가장 유한 공간의 색깔이 (다른 공간 색과 적절히 융합(∑)되어) 계산되어 나오겠다. 정답값인 C_j와 비교해서 loss 도출.
    - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210115143836240.png?raw=tru" alt="image-20210115143836240" style="zoom: 67%;" />
    - 괜찮은 colorization 성능 도출 되었다. 
    - 이렇게 학습시킨 모델을 사용해서, 아래의 문제에서도 좋은 결과를 얻을 수 있었다.  단! colorization과 비슷하고 Reference Frame(Ex) previousframe, previous pose은 항상 제공된다.
    - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210115144026487.png?raw=tru" alt="image-20210115144026487" style="zoom: 50%;" />



## 2.Part (3) - Videos with Sound

- Key : Sound and Frames are Semantically consistent and Synchronized. (일관적인 관계가 있다.)
- Objective(목적): use vision and sound to learn from each other (서로서로를 예측하기 위해 서로를 사용한다.)
- Audio-Visual Embedding (AVE-Net)
  - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210115144930147.png?raw=tru" alt="image-20210115144930147" style="zoom:67%;" />
  - output : Positive or Negative. 
  - Results Audio features : Sound Classification (ESC-50 dataset) 에서 가장 좋은 성능을 얻어냄. 
  - Results Visual features : ImageNet images를 학습하지 않아도 좋은 성능 발현 됨.
- AVOL-Net
  - 사진에서 어디서 소리가 나는지 예측.
  - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210115150920684.png?raw=tru" alt="image-20210115150920684" style="zoom: 67%;" />
  - input: audio and video frame
  - Output: localization heatmap on frame
- Other papers.. : Ambient sound provides supervision for visual learning / Visually indicated sounds
- DataSet : AudioSet from Youtube
- 지금까지 소리와 이미지의 상관성(correspondence)를 파악하고자 노력했다. 이제는 이미지와 소리와의 synchronization을 파악하는 노력을 해보자.
  - 우리는 입모양으로 그 사람이 뭐라고 말하는지 유추가 가능하다. 시각장애인들은 더더욱.
  - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210115151120063.png?raw=tru" alt="image-20210115151120063" style="zoom: 80%;" />
  - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210115151510905.png?raw=tru" alt="image-20210115151510905" style="zoom:67%;" />
  - Lip Synchronization, Active speaker detection 에서 더 좋은 성능을 내었다. 



## 2.3 - Summary

1. Self-supervised learning from images/video
   - without explicit supervision, 학습이 가능하다. (Pre-train모델을 만들 수 있었다.)
   - Learns visual representations = 적절한 Visual Feature Extractor들을 뽑을 수 있었다.
2. Self-Supervised Learning from videos with sound
   1. Intra- and cross-modal retrieval - 서로의 데이터로 상호 학습이 가능했다.
   2. Learn to localize sounds (어디서 소리가 나는지 예측하는 모델)
   3. (우리가 연구해 볼 수 있는)**다른 domain pair는?** 
      - face(not lip) - voice 
      - **Infrared(적외선) - visible** 
      - RGB - Depth 
      - Stereo streams


# 3. Self-Supervised Representation Learning

- [Link](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html)
- 나중에 필요하면 공부해보자. 우선 위의 내용으로 self-supervised Learning에 대해서 큰 그림을 잡을 수 있었다.



























