---
layout: post
title: 【Detection】Understanding Cascade R-CNN paper with code 
---

- **논문** : [Cascade R-CNN: Delving into High Quality Object Detection](https://openaccess.thecvf.com/content_cvpr_2018/papers/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.pdf)

- **분류** : Object Detection

- **저자** : Zhaowei Cai, Nuno Vasconcelos

- **읽는 배경** : Recognition Basic. Understand confusing and ambiguous things.

- **읽으면서 생각할 포인트** : 코드와 함께 최대한 완벽히 이해하기. 이해한 것 정확히 기록해두기.

- **느낀점**  : 

  - 최근 논문일수록, 블로그나 동영상의 내용 정리가 두리뭉실하고 이해하기 힘들다. 디테일성은 거의 없다. 따라서 그냥 논문 읽는게 최고다. 만약 블로그에 내용 정리가 잘 되어 있다고 하더라도, 내가 블로그 내용에 대한 신뢰가 안가서, 합리적 추론의 이해(덜 디테일한 설명은 나의 생각을 좀 더 추가해서 이해하는 과정)을 할 수도 없다. 따라서 **논문이나 읽자. 시간 낭비하지 말고.**
  - SSD 코드를 공부했을때 모듈화가 심각해서 보기가 힘들었다. 하지만 그것은 "처음부터 끝까지 다 봐야지." 라는 욕심때문에 보기 힘들었던 것 같다. 하지만 사실 그렇게 코드를 보는 경우는 드믄것 같다. "내가 궁금한 부분만 찾아보거나, 내가 사용하고 싶은 모듈만 찾아서 사용한다."라는 마음으로 부분 부분 코드를 본다면, 내가 원하는 부분을 전체 코드에서 찾는 것은 그리 어렵지 않다. 라는 것을 오늘 느꼈다. 
  
  



# 1. Cascade R-CNN

## 1-(1). 참고 자료 

1. (1) [Cascade R-CNN 블로그](https://blog.lunit.io/2018/08/13/cascade-r-cnn-delving-into-high-quality-object-detection/), (2) [Cascade R-CNN Youtub동영상1](https://www.youtube.com/watch?v=1_-HfZcERJk&feature=youtu.be)

2. 참고 자료 소감 : 

   - 블로그 : 너무 result 해석에 많은 초점을 둔다. 방법론에 대해서는 구체적이지는 않아, 나에게는 이해가 어려웠다. 논문이나 읽자.
   - 동영상 : 핵심이 Casecade-Mask RCNN이다. 디테일이 거의 없다. 안봄.

3. 참고자료 내용 정리 : 

   1. 이 논몬은 object detector의 약점을 잘 파악했다. 이 논문을 통해, 아주 복잡한 방법을 적용해서 성능 향상을 미약하게 이루는 것 보다는, **문제점을 잘 파악하기만 하면 어렵지 않게 성능을 향상시킬 수 있음을 보여준다.**

   2. 문제점 파악하기

      1. Introduction에 실험 그래프가 나온다. 블로그 글은 이해 안되서, 논문 읽는게 낫겠다.    

         <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210204160324773.png?raw=tru" alt="image-20210204160324773" style="zoom: 80%;" />

   3.  Cascade R-CNN은 위 사진의 오른쪽 그림의 구조이다. 굳이 설명하지 않겠다. 특히 단계를 거듭할 수록 **보다 더 높은 IOU를 기준으로 학습시킨다.**

      - 과거 방식 (1) 단순하게 하나의(같은) classifier를 Iterative하게 사용하는 것은 큰 성능 향상을 주지 못한다.  (2) 서로 다른 classifier를 여러개 만들고, 각각의 IOU기준을 다르게 주고 학습을 시키는 방법도 성능 향상은 그저 크지 않다. 즉 여러 classifier의 ensenble 방법이다.
      - Cascade R-CNN은 각각의 classifier는 각각 Threshold 0.5, 0.6, 0.7 로 학습. 예측 bounding box와 GT-box가 겹치는 정도가 Threshold 이상이여야지만, 옳게 예측한 것이라고 인정해줌. 이하라면 regressing 틀린거로 간주하고 loss를 준다. 



## 1-(2). Paper Review 

1. **Conclustion**	
   - multi-stage object detection framework (=extension)
   - overfitting, mismatch at inference 문제점 해결
2. **기본 이론**
   - Bounding Box와 GT-box와의 **IOU threshold**보다...
     - 크면 **Positive** Bounding box(당당하게 이건 객체다! 라고 말하는 예측 BB값), 
     - 작으면 **Negative**(Bounding Box이긴 하지만, 아무래도 이건 객체는.. 아니다. 라고 말하는 예측 BB값)
     - 여기서 P, N이, [TP FP TN FN]에서 P, N 이다. 
3. **abstract, Introduction**
   - 현재 문제점들 
     - close false positives = “close but not correct 거의 맞췄지만.. 아니야" bounding boxes 가 많기 때문에 더욱 어려운 문제이다.
     - 통상적으로 우리는 IOU threshold = 0.5 를 사용합니다. 하지만 이것은 a loose requirement for positives (약간은 느슨한,쉬운 요구조건) 입니다. 따라서 detector는 자주 noise bounding box를 만들어 낸다. 이것 때문에 우리는 항상 reject close false positives 에 어려움을 더 격고 있다. 
     - 즉. 아래의 그림과 같이, u=0.5일 때.. Detected bounding box에 대해서 너무 약한 threshold를 주기 때문에 아래와 같은 noise스러운 박스들이 많이 추출 된다.    
       <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210204172936186.png?raw=tru" alt="image-20210204172936186" style="zoom:70%;" />
   - 분석해보기
     - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210204173847031.png?raw=tru" alt="image-20210204173847031" style="zoom:70%;" />
     - 그림(c) : X : Input Proposal(RPN에서 나오는 BB와 GT), Y : detection performance(Classifier에서 오는 BB와 GT) u=0.7에서는 일정 이상의 X에서만 높은 Y값이 나온다. 
     - 그림(d) : u=0.5로 했을때(0.5이상은 모두 객체로 탐지. 모든 Positive들), Threshold(이 이상만 TP로 판단. 이 이외는 FP)가 어떤 때든 상대적으로 좋은 AP가 도출 된다. 
     - ( 일반적으로 이렇게 하나의 IOU값으로 최적화된 모델(Detector)은 다른 IOU threshold에서도 최적화된 값(Hypotheses=예측/추론값)이 나오기 힘들다. )
     - 그렇다고 u를 그냥 높게 하는 건 좋지 않다. (d)에서 보이는 결과도 그렇고 Overfitting이 발생하는 경향도 있다. 또한 (c)에서 처럼 u=0.7에서는 일정 이상의 X에서만 높은 Y값이 나온다. 일정 이하는 그리.. 최적화된 신경망이 만들어 진다고 볼 수 없다.
     - 이 논문의 저자는 생각했다.  bounding box regressor를 0.5 ~ 0.6 ~ 0.7 의 u로 키워가며 stage를 거치게 하는 것은 (c)에서, 낮은 X에서는 파랑Line을, 중간 X에서는 초록Line을, 높은 X에서는 빨간Line이상의 값을 OutputIOU로 가지게 함으로써 적어도 회색 대각선보다는 높은 곳에 점이 찍히게 하는 것을 목표로 한다. 
   - 우리의 방법
     - **나만의 직관적 이해 :** ⭐⭐ 
       1. 만약 RPN에서 나온 BB와 GT간의 IOU가 0.55이고, Classifier에서 나온 BB와 GT간의 IOU가 0.58라고 해보자.(class는 맞았다고 가정) 
       2. 만약 Threshold=0.6이라면, (클래스는 맞았음에도 불구하고 BB위치가 틀렸기 때문에) 이것은 FP라고 처리 된다. 
       3. 이것이야 말로 close false positives 이다. 
       4. 하지만 아까 0.58의 BB정보를 sequentially하게 옆의 Classifier에 넘겨주고 이 Classifier에서 나오는 BB와 GT간의 IOU가 0.61이 된다면! 
       5. 이것은 완벽히  close false positives에 대한 문제를 해결해준 것이라고 할 수 있다. 
     - Cascade R-CNN does not aim to mine hard negatives.(즉 클래스에 관점에서 AP를 높히는 것이 아니다.) Instead, by adjusting bounding boxes(Regressing 관점에서 AP를 높이려는 목표를 가진다.)
4. **Related Work,  Object Detection**
   1. <u>Offset Relative 공식</u> :    
      <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210204220629271.png?raw=tru" alt="image-20210204220629271" style="zoom:120%;" />
   2. ![image-20210204220646554](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210204220646554.png?raw=tru)
   3. Figure 3 (b) : same classifier / 모두 u = 0.5 / 따라서 위의 그래프 (c)에서 보는 것처럼 u=0.5이고 X가 일정 이상일때 오히려 Y값이 더 낮아지는 현상을 간과 했다. /  또한 위 그림의 Figure2에서 보는 것처럼, (blue dot) 첫번째 stage에서 BB의 offset Relative로 인해 refine이 이뤄지지만, 두, 세번째 stage에서는 분산 값이 좁게 변한다. 초기에 이미 Optimal이라고 판단하는 듯 하다 /
   4. <u>Detection Quality</u> 
      - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210204222136796.png?raw=tru" alt="image-20210204222136796" style="zoom: 80%;" />
   5. <u>Figure3-(c) Integral Loss</u>
      - U = {0.5, 0.55, ⋅⋅⋅ , 0.75}를 가지는 Classifier들을 모두 모두 모아서 Ensemble을 수생한다. 
      - 많은 Classifier가 있으므로, 이들에 대한 모든 Classifier Loss는 아래와 같다.
      - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210204222421487.png?raw=tru" alt="image-20210204222421487" style="zoom: 67%;" />
      - 아래 Figure4의 첫번째 그래프를 보더라도, 0.7이상의 IOU를 가지는 BB는 2.9프로 밖에 안된다. 이래서 좋은 U를 가지는 Classifier는 빨리 Overfitting이 되버린다. 따라서 이러한 방법으로  close false positives문제가 해결되지는 않았다. 
      - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210204222640714.png?raw=tru" alt="image-20210204222640714" style="zoom: 60%;" />
5. **Cascade R-CNN**
   1. <u>Cascaded Bounding Box Regression</u>
      - 4-3에서 blue dot에 대한 설명과 다르게 distribution이 1-stage와 비슷하게 유지되도록 노력했다. (그래서 나온 결과가 u=0.5, 0.6, 0.7)
      - ![image-20210204232921641](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210204232921641.png?raw=tru) 는 normalized 된다.   [Faster R-CNN] 널리 사용되는 방법이란다. 코드에서는 구체적인 normalization을 이루진 않았다. (필요하면! 아래의 아래 6-2 stat  부분 참조.)
   2. <u>Cascaded Detection</u>
      - ![image-20210204234748605](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210204234748605.png?raw=tru)
6. **Experimental Results**
   1. <u>Implementation Details</u>
      - four stages :  1_RPN and 3_detection with U = {0.5, 0.6, 0.7}
      - horizontal image flipping 이외에 어떤 data augmentation도 사용되지 않았다.
   2. <u>Generalization Capacity</u>
      - 지금까지는 예측 BB의 GT와의 IOU를 증가시키려고 노력했다. 그럼 Figure(3)-d에서 C1,C2,C3는 어떻게 이용할까? 실험을 통해서 그냥 C3를 이용해서 Class 예측을 하는 것보다는, C1,C2,C3을 **앙상블**함으로써 더 좋은 AP를 얻어내었다고 한다. 
      - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210205123349693.png?raw=tru" alt="image-20210205123349693" style="zoom:80%;" />
      - (블로그 내용 참조 그리고 ~~FasterRCNN논문도 나중에 참고해보기.~~ 참고해봤는데 이런 내용 없다. 그래서 일단 블로그 작성자에게 "구체적으로 어디서 나온 내용이냐고" 질문 해봄. 질문이 잘 갔는지 모르겠다...) Regression Statistics = stat : 위의 offset relative 델타 값들은, Faster-RCNN에서 이 파라메터가 잘 나오게 학습시킬 때 L1 loss를 사용한다. 이때 델타값을 normalization하지 않으면 자칫 학습이 잘 이뤄지지 않는 문제가 있었다고 한다. 그래서 각 좌표값을 normalize하는 작업을 수행했다. (4-1의 수식처럼) 각 Stage에서 각자의 기준으로, regressing offset relative값에 대한 normalization을 수행했다는 의미인듯 하다.
      - Cascade 모듈은 다른 곳에서 쉽게 적용될 수 있다. 다른 기본 모델에 cacade모듈을 적용하여 효과적인 AP 상승을 이뤄낼 수 있었다고 한다. 



# 2. guoruoqian/cascade-rcnn_Pytorch

1. Github Link : [guoruoqian/cascade-rcnn_Pytorch](https://github.com/guoruoqian/cascade-rcnn_Pytorch)

2. 코드가 어느 정도 모듈화가 되어 있다. 하지만 보는데 그리 어려운 정도는 아니다. 

3. ROI pooing + ROI Align/ Cascade + Noncacade/ 와 같이 비교를 위한 코드도 함께 들어가 있다.

4. 핵심 신경망 구현 코드는 이것이다. [cascade-rcnn_Pytorch/lib/model/fpn/cascade/fpn.py](https://github.com/guoruoqian/cascade-rcnn_Pytorch/blob/master/lib/model/fpn/cascade/fpn.py)    
   여기서 FPN도, RPN의 사용도, Cacade Multi-stage Module, Loss계산 까지 모두 구현 되어있다. 물론 다른 곳에서 구현된 함수와 클래스를 마구마구 사용하지만....

5. ```python
   # cascade-rcnn_Pytorch/lib/model/fpn/cascade/fpn.py
   class _FPN(nn.Module):
       
       def __init__(self, classes, class_agnostic):
           self.RCNN_loss_cls = 0
           self.RCNN_loss_bbox = 0
           self.RCNN_rpn = _RPN_FPN(self.dout_base_model)
           self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
           # 다른 신경망을 봐야함 FPN에는 없음 아래 내용 2nd, 3rd에 동일 적용
           self.RCNN_bbox_pred = nn.Linear(1024, 4)
           self.RCNN_bbox_pred = nn.Linear(1024, 4 * self.n_classes)
           # 이게 u = 0.5, 0.6, 0.7 적용은 아래의 신경망이 사용된다.
           self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
           
       def forward(self, im_data, im_info, gt_boxes, num_boxes):
   		# Feature Map 뽑기
           rpn_feature_maps = [p2, p3, p4, p5, p6]
           mrcnn_feature_maps = [p2, p3, p4, p5]
           rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(rpn_feature_maps, im_info, gt_boxes, num_boxes)
           
           # 첫번째 Classifier
           roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
           roi_pool_feat = self._PyramidRoI_Feat(mrcnn_feature_maps, rois, im_info)
           pooled_feat = self._head_to_tail(roi_pool_feat)
           bbox_pred = self.RCNN_bbox_pred(pooled_feat)
           cls_score = self.RCNN_cls_score(pooled_feat)
           # 두번째 Classifier
           roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes, stage=2)
           roi_pool_feat = self._PyramidRoI_Feat(mrcnn_feature_maps, rois, im_info)
           pooled_feat = self._head_to_tail_2nd(roi_pool_feat)
           bbox_pred = self.RCNN_bbox_pred_2nd(pooled_feat)
           cls_score = self.RCNN_cls_score_2nd(pooled_feat)
           # 세번째 Classifier
           roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes, stage=3)
           roi_pool_feat = self._PyramidRoI_Feat(mrcnn_feature_maps, rois, im_info)
           pooled_feat = self._head_to_tail_3rd(roi_pool_feat)
   		bbox_pred = self.RCNN_bbox_pred_3rd(pooled_feat)
           cls_score = self.RCNN_cls_score_3rd(pooled_feat)
               
   ```

6. ```python
   self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
   
   # cascade-rcnn_Pytorch/lib/model/rpn/proposal_target_layer.py 
   class _ProposalTargetLayer(nn.Module):
       """
       Assign object detection proposals to ground-truth targets. 
       Produces proposal classification labels and bounding-box regression targets.
       
       내가 예측한 BB값과 GT값을 비교해서, Positive에 해당하는 BB만을 return하는 코드 같다
       """
       if stage == 1:
           fg_thresh = cfg.TRAIN.FG_THRESH
           bg_thresh_hi = cfg.TRAIN.BG_THRESH_HI
           bg_thresh_lo = cfg.TRAIN.BG_THRESH_LO 
       elif stage == 2:
           fg_thresh = cfg.TRAIN.FG_THRESH_2ND
           bg_thresh_hi = cfg.TRAIN.FG_THRESH_2ND
           bg_thresh_lo = cfg.TRAIN.BG_THRESH_LO
       elif stage == 3:
           fg_thresh = cfg.TRAIN.FG_THRESH_3RD
           bg_thresh_hi = cfg.TRAIN.FG_THRESH_3RD
           bg_thresh_lo = cfg.TRAIN.BG_THRESH_LO
       ... 
      
   # cascade-rcnn_Pytorch/lib/model/utils/config.py 
   __C.TRAIN.FG_THRESH = 0.5
   __C.TRAIN.FG_THRESH_2ND = 0.6
   __C.TRAIN.FG_THRESH_3RD = 0.7
   ```

7. 델타 (offset relative) normalization  : 확실한 건 아니지만, 아래의 코드에 이런 것을 확인할 수 있었다. 하지만   아래의 과정은 rois의 x,y값을 그저 0~1사이의 값으로 만들어 줄 뿐이다. 따라서 normalization는 아니다. // 그 아래의 코드에 BBOX_NORMALIZE_STDS라는 것도 있는데, 이건 GT에 대해서 normalization을 하는 코드이다. // 아무래도 역시 Faster-RCNN에서 먼저 공부하고 다른 Faster-RCNN이나 Mask-RCNN의 코드를 함께 보는게 더 좋겠다.

   ```python
   if self.training:
       roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
       rois, rois_label, gt_assign, rois_target, rois_inside_ws, rois_outside_ws = roi_data
   
       ## NOTE: additionally, normalize proposals to range [0, 1],
       #        this is necessary so that the following roi pooling
       #        is correct on different feature maps
       # rois[:, :, 1::2] /= im_info[0][1] # image width로 나눔
       # rois[:, :, 2::2] /= im_info[0][0] # image height로 나눔
   
   # lib/model/rpn/proposal_target_layer.py 
   class _ProposalTargetLayer(nn.Module):
       def __init__(self, nclasses):
            self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
            self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)
       def forward(self, all_rois, gt_boxes, num_boxes, stage=1):
           self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
           self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
           self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)
           
   ```

   
   

































