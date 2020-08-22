---
layout: post
title: (위성Segment) Segmentation Survey 논문 작성 목표로, 읽을 논문 정리
description: >  
    
---
(위성Segment) Segmentation Survey 논문 작성 목표로, 읽을 논문 정리
## 1. Awesome segmentation
(1) Paper & Git  
{:.lead}

1. [Awesome semantic segmentation](https://github.com/mrgloom/awesome-semantic-segmentation) : (6.2k stars) semantic segmentation 논문들 모두 정리 + Git 코드 정리
2. [Awesome-segmentation](https://github.com/manhcuogntin4/awesome-segmentation) : (8 stars) semantic segmentation은 위의 사이트와 동일. instance segmentation도 추가 되어 있음. 
3. [Awesome-satellite-segmentation](https://awesomeopensource.com/project/chrieke/awesome-satellite-imagery-datasets) : (1.3k stars)  List of satellite image training datasets 
4. [Really-awesome-semantic-segmentation](https://github.com/nightrome/really-awesome-semantic-segmentation) : semantic-segmentation에 관한 Survey papers도 있다.  

(2) Code  
{:.lead}

1. [awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch) : (1k stars)Semantic Segmentation과 관련된 (살짝 과거) 기술들을 모아서 코드화 시켜놓았다. 다 읽고 정독하면 좋을 듯 하다. 
   On PyTorch (include FCN, PSPNet, Deeplabv3, Deeplabv3+, DANet, DenseASPP, BiSeNet, EncNet, DUNet, ICNet, ENet, OCNet, CCNet, PSANet, CGNet, ESPNet, LEDNet, DFANet) 
2. [semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch) : (3.1k stars) 최근 Segmentation 기술들을 코드화 시켜놓았다. Pytorch implementation for Semantic Segmentation/Scene Parsing on MIT ADE20K dataset





## 2. Survey paper - segmentation
(1) 위의 Really-awesome-semantic-segmentation 에 있는 논문들  
{:.lead}

여기 있는 논문들은 **semantic-segmentation를 어디에 사용했는가** 가 중심인듯 하다.

- 2018 (Cite:49) [RTSeg: Real-time Semantic Segmentation Comparative Study](https://arxiv.org/abs/1803.02758)  

- 2018 (Cite:13) [Indoor Scene Understanding in 2.5/3D: A Survey](https://arxiv.org/abs/1803.03352) 

- 2017 (Blog) [A 2017 Guide to Semantic Segmentation with Deep Learning by Qure AI](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review)  

- 2017 (Cite:455) [A Review on Deep Learning Techniques Applied to Semantic Segmentation](https://arxiv.org/abs/1704.06857) 

- <u>2017 (Cite:179)</u> [Computer Vision for Autonomous Vehicles: Problems, Datasets and State-of-the-Art](https://arxiv.org/abs/1704.05519) [[Webpage\]](http://www.cvlibs.net/projects/autonomous_vision_survey/)  

(2) Google Scholar  
{:.lead}

- **2020 (cite:6) [Image Segmentation Using Deep Learning: A Survey](https://arxiv.org/abs/2001.05566)** 
- **2019 (cite:41) [Survey on semantic segmentation using deep learning techniques](https://www.sciencedirect.com/science/article/pii/S092523121930181X)** 
- <u>2019 (cite:10)</u> [A Brief Survey and an Application of Semantic Image Segmentation for Autonomous Driving](https://arxiv.org/abs/1808.08413)
- 2018 (cite:93) [A survey on deep learning techniques for image and video semantic segmentation](https://www.sciencedirect.com/science/article/pii/S1568494618302813)
- 2017 (cite:6) [Survey on semantic image segmentation techniques](https://ieeexplore.ieee.org/document/8389420) 

(3) DBpia  
{:.lead}

  - 한글 Survey(조사) 논문은 없다. 

    



## 3. Survey paper - satellite segmentation
(1) satellite segmentation 대회 주요 논문  
{:.lead}

- Kaggle 대회 논문(2017) : [Satellite Imagery Feature Detection: A Kaggle Competition](https://arxiv.org/abs/1706.06169)
- Codalab 대회 논문(2018): [DeepGlobe 2018: A Challenge to Parse the Earth through Satellite Images](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Demir_DeepGlobe_2018_A_CVPR_2018_paper.pdf)

(2) satellite survey 논문  
{:.lead}

- 논문이 없다. 

(3) satellite segmentation 연구 논문  
{:.lead}

- 2016 (90p-cite:12) [Semantic Segmentation of Satellite Images using Deep Learning](http://www.diva-portal.org/smash/get/diva2:1013270/FULLTEXT01.pdf)





## 4. PS
1. 논문 읽는 방법 : [논문 읽는 방법](https://woongheelee.com/entry/%EB%85%BC%EB%AC%B8%EC%9D%84-%ED%9A%A8%EC%9C%A8%EC%A0%81%EC%9C%BC%EB%A1%9C-%EC%9D%BD%EB%8A%94-%EB%B0%A9%EB%B2%95) 
   - 하나의 논문은 3 pass 과정을 거쳐서 3번은 훑어야한다. 
     - 논문의 전반적인 아이디어 이해
     - 디테일을 제외한 논문의 내용 이해
     - 깊은 이해
   - Literature Survey

**step1**  
{:.lead}  
제목, abstract, introduction, 각 섹션의 제목, Conclusion

**step2**  
{:.lead}   
논문에 더욱 집중해서 읽어라  
증명과 같은 세세한 것들은 무시해라  
핵심을 써내려가라  
그림, 다이어그램, 그리고 다른 삽화들을 주의 깊게 살펴보아라. 특히나 그래프에 신경을 써서 보아라  

**step3**  
{:.lead}  
가장 중요한 것은 논문을 가상으로 재 실험해보는 것이다  

