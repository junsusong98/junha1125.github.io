---
layout: post
title: 【Detection】Understanding SSD paper with code 
---

- **논문** : [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf)
- **분류** : Original Object Detection
- **저자** : Wei Liu , Dragomir Anguelov, Dumitru Erhan , Christian Szegedy
- **읽는 배경** : Recognition Basic. Understand confusing and ambiguous things.
- **읽으면서 생각할 포인트** : 코드와 함께 최대한 완벽히 이해하기. 이해한 것 정확히 기록해두기.
- **느낀점** : 



# 0. Abstract

- 









# mmdetection for SSD

1. `init_detector`를 통해서 SSD inference 되는 과정
   1. def **init_detector**(config, checkpoint=None, device='cuda:0', cfg_options=None):
   2. model = **build_detector**(config.model, test_cfg=config.get('test_cfg'))
   3. def **build_detector**(cfg, train_cfg=None, test_cfg=None):
   4.  return **build**(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
   5. def **build**(cfg, registry, default_args=None):
   6. return **build_from_cfg**(cfg, registry, default_args)
   7. from **mmcv**.utils import Registry, **build_from_cfg**
   8. def **build_from_cfg**(cfg, registry, default_args=None):
   9. 후.. 여기까지. 일단 패스
2. 아래의 과정을 통해 Config과정을 통해서 SSD가 이뤄지는지 보기 위한 작업들이다.    
   ![image-20210130153957160](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210130153957160.png)
3. 위의 결과를 통해 나는 이런 결과를 낼 수 있었다. **<u>mmdetection을 가지고 모델 내부 구조 코드를 볼 생각을 하지 말자.</u>** 



