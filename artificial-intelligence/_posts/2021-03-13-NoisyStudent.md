---
layout: post
title: 【Detection】[Self-training with Noisy Student improves ImageNet classification
---

- **논문** : [Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/abs/1911.04252)
- **분류** : classification (Detection)
- **저자** : Qizhe Xie, Minh-Thang Luong, Eduard Hovy
- **느낀점** 
- **목차**
  1. Paper Review
  2. Noise 기법 정리



# Self-training with Noisy Student

# 1. Conclusion, Abstract

1. 과거의 기법들이, ImageNet에서의 성능 향상을 위해서, 수십억장의 web-scale extra labeled images와 같은 많은 `weakly labeled Instagram images` 이 필요한 `weakly-supervised learning` 을 수행했었다.
2. 하지만 우리는 `unlabeled images`을 사용함으로써, 상당한 성능향상을 얻어내었다. 즉 the student에게 Nosie를 추가해 줌으로써, 성능향상을 이루는, `self-training = Noisy Student Training = semi-supervised learning` 기법을 사용하였다. 
3. EfficientNet에 Noisy Student Training를 적용함으로써 accuracy improvement와 robustness boost를 획득했다. 
4. 전체 과정은 다음과 같다.
   1. 우선 Labeled Image를 가지고 EfficientNet Model(E)을 학습시킨다. 
   2. E를 가지고 300M개의 unlabeled images에 대해서 Pseudo labels를 생성한다. 
   3. E를 Teachear로 사용한다. 
   4. **larger EfficientNet**를 student model로 간주하고, labeld + sseudo labeld images를 사용해 학습시킨다. 
   5. student의 학습을 진행하는 동안, **dropout[76], stochastic depth[37], RandAugment data augmentation[18]** 와 같은 noise를 주입한다. 
   6. teacher보다 더 똑똑한 student가 탄생한다!

![image-20210313180644542](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210313180644542.png)



---

# 2. Noisy Student Training







---

---

# Noise 기법 정리

1. **Dropout** [2014]     
   <img src="C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210313170600697.png" alt="image-20210313170600697" style="zoom: 80%;" />
   - overfitting 방지
   - hidden unit을 일정 확률로 0으로 만드는 regularization 기법이다. 
   - 후속 연구로, connection(weight)를 끊어버리는 (unit은 다음층 다른 unit과 모두 연결되어 있는데, 이 중 일부만 끊어 버리는 것이다. dropout 보다는 조금 더 작은 regularization(규제)라고 할 수 있다. )

2. **stochastic depth** [2016]
   - ResNet의 layer 개수를 overfitting 없이 크게 늘릴 수 있는 방법이다. ResNet1202 를 사용해도 정확도가 오히려 떨어지는 것을 막은 방법이다.
   - ResNet에 있어서 훈련할 때에 residual 모듈 내부를 랜덤하게 drop(제거)하는 모델이다. (모듈 내부가 제거되면 residual(=shortcut)만 수행되며, 그냥 모듈 이전의 Feature가 그대로 전달되는 효과가 발생한다.)
   - Test시에는 모든 block을 active하게 만든 full-length network를 사용한다.
   - p_l = 1 - l/2L 
     - residual 모듈이 drop하지 않고 살아남을 확률이다. 
     - L개의 residual 모듈에서 l번째 모듈을 의미한다. 
     - input에 멀어질 수록, l은 커지고, p_l은 작아진다. 즉 drop 될 확률이 커진다.
   - p_l의 확률값에 의해서 b_l (0 or 1 = drop_active or Non_active)이 결정된다.     
     ![image-20210313171503869](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210313171503869.png)

3. **RandAugment data augmentation** [2019]
   - 간략하게 말하면, 기존에 다양한 Data Augmentation을 싹 정리해놓은 기법이다. 
   - 아래와 같은 14가지 Augmetation 기법들을 모아놓고, 랜덤하게 N개를 뽑고, 얼마나 강하게 Transformation(distortion magnitude)를 줄 것인지 M (Magnitude)를 정해준다. (아래 왼쪽의 수도코드 참조)
   - 그렇다면 M을 얼마나 주어야 할까? (1) 매번 랜덤하게 주는 방법 (2) 학습이 진행될수록 키우는 방법 (3) 처음부터 끝까지 상수로 놔두는 방법 (4) 상한값 이내에서 랜덤하게 뽑되, 상한가를 점점 높히는 방법
   - 모두 실험해본 결과! 모두 같은 성능을 보였다. 따라서 가장 연산 효율이 좋은 (3)번을 사용하기로 했고, 상수 값 M을 몇으로 하는게 가장 좋은 성능을 내는지 실험해 보았다. (아래 오른쪽 그래프 참조) 그래프 분석에 따르면, 최적의 M은 10~15 정도인 것을 알 수 있다.  
     <img src="C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210313173515056.png" alt="image-20210313173515056" style="zoom:80%;" />   