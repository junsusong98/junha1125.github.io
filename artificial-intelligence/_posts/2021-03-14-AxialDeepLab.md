---
layout: post
title: 【Pa-Segmen】Axial-DeepLab
---

- **논문** : [Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation](https://arxiv.org/abs/2003.07853)
- **분류** : Panoptic Segmentation
- **저자** : Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam
- **느낀점** 
- **목차**
  1. Axial-DeepLab Paper Review
  2. Youtbe 강의 내용 정리



# Axial-DeepLab

# 1. Conclusion, Abstract

- `position-sensitive, axial attention, without cost`이 Classification과 Segmentation에서 얼마나 효율적인지를 보여주었다.
- Convolution은 ` long range context`를 놓치는 대신에 `locality attention`을 효율적으로 처리해왔다. 그래서 최근 work들은, `local attention`을 제한하고, `fully, global attention` 을 추가하는 self-attention layer 사용해왔다. 
- 우리는 `fully, stand-alone, axial attention`은  2D self-attention을 `1D self-attention x 2개`로 분해하여 만들어 진 self-attention 모듈이다. 이로써 `large & global receptive field`를 획득하고, complexity를 낮추고, 높은 성능을 획득할 수 있었다.
- introduction, Related Work는 일단 패스



---

# 3. Method

- Key word :  (1) `position-sensitive self-attention`, (2) `axial-attention`, (3) `stand-alone Axial-ResNet` (4) ` Axial-DeepLab`
- 





---

---

# youtube 내용 정리

- Youtube Link : [https://www.youtube.com/watch?v=hv3UO3G0Ofo](https://www.youtube.com/watch?v=hv3UO3G0Ofo)

1. Intro & Overview 
   - transformer가 NLP에서 LSTM을 대체한 것 처럼, Image process에서도 convolution을 대체할 것이다. 이러한 방향으로 가는 step이 이 논문이라고 할 수 있다.           
2. This Paper's Contributions 
3. From Convolution to Self-Attention for Images 
4. Learned Positional Embeddings 
5. Propagating Positional Embeddings through Layers 
6.  Traditional vs Position-Augmented Attention 
7. Axial Attention 
8. Replacing Convolutions in ResNet 
9. Experimental Results & Examples

