---
layout: post
title: 【LightWeight】Understanding EfficentNet+EfficentDet paper w/ code
---

- **논문1** : [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf) / [Youtube Link](https://www.youtube.com/watch?v=Vhz0quyvR7I)
- **논문2** : [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070) / [Youtube Link](https://www.youtube.com/watch?v=11jDC8uZL0E)
- **분류** : LightWeight
- 저자 : Mingxing Tan, Ruoming Pang, Quoc V. Le
- **읽는 배경** : Recognition Basic. Understand confusing and ambiguous things.
- **읽으면서 생각할 포인트** : 코드와 함께 최대한 완벽히 이해하기. 이해한 것 정확히 기록해두기.
- **느낀점**  : 
- **목차**
  
  1. EfficientNet from youtube ([바로가기](https://junha1125.github.io/blog/artificial-intelligence/2021-02-10-EfficentNet/#1-efficientnet-from-youtube))
  3. Code - lukemelas/EfficientNet-PyTorch ([바로가기](https://junha1125.github.io/blog/artificial-intelligence/2021-02-10-EfficentNet/#2-lukemelasefficientnet-pytorch))
  4. EfficientDet from youtube ([바로가기](https://junha1125.github.io/blog/artificial-intelligence/2021-02-10-EfficentNet/#3-efficientdet-from-youtube))
  6. Code - zylo117/Yet-Another-EfficientDet-Pytorch ([바로가기](https://junha1125.github.io/blog/artificial-intelligence/2021-02-10-EfficentNet/#4-zylo117yet-another-efficientdet-pytorch))
  





# 1. EfficientNet from youtube

![img01](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-01.png?raw=true)
![img02](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-02.png?raw=true)
![img03](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-03.png?raw=true)
![img04](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-04.png?raw=true)
![img05](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-05.png?raw=true)
![img06](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-06.png?raw=true)
![img07](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-07.png?raw=true)
![img08](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-08.png?raw=true)
![img09](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-09.png?raw=true)
![img10](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-10.png?raw=true)
![img11](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-11.png?raw=true)
![img12](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-12.png?raw=true)
![img13](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-13.png?raw=true)
![img14](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-14.png?raw=true)
![img15](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-15.png?raw=true)
![img16](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-16.png?raw=true)
![img17](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-17.png?raw=true)
![img18](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-18.png?raw=true)
![img19](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-19.png?raw=true)
![img20](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-20.png?raw=true)
![img21](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-21.png?raw=true)
![img22](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-22.png?raw=true)
![img23](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-23.png?raw=true)
![img24](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-24.png?raw=true)
![img25](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-25.png?raw=true)
![img26](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-26.png?raw=true)
![img27](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-27.png?raw=true)
![img28](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-28.png?raw=true)
![img29](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-29.png?raw=true)
![img30](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-30.png?raw=true)
![img31](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-31.png?raw=true)
![img32](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientnet_youtube/efficientnet_youtube-32.png?raw=true)




# 2. lukemelas/EfficientNet-PyTorch

- Github-Link : [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)

- EfficientNet은 a family of image **classification** models 이다. Based on **MnasNet** in term of **AutoML**, **Compound Scaling**.

- Simply, Model 불러와 Classification 수행하기

  - ```python
    import json
    from PIL import Image
    import torch
    from torchvision import transforms
    
    from efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b0')
    
    # 0. Preprocess image
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    img = tfms(Image.open('img.jpg')).unsqueeze(0)
    print(img.shape) # torch.Size([1, 3, 224, 224])
    
    # 0. Load ImageNet class names
    labels_map = json.load(open('labels_map.txt'))  # 이미 EfficientNet-Pytorch/examples/simple/labels_map.txt 있다.
    labels_map = [labels_map[str(i)] for i in range(1000)]
    
    # 1. For Classify
    model.eval()
    with torch.no_grad():
        outputs = model(img)
    
    # 2. For Feature Extractor
    features = model.extract_features(img)
    print(features.shape) # torch.Size([1, 1280, 7, 7])
    ```
  
- Pytorch Efficient-Net model Code 

  - [efficientnet_pytorch/model.py](https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py) 코드 참조





---

---



# 3. EfficientDet from youtube

![img01](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-01.png?raw=true)
![img02](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-02.png?raw=true)
![img03](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-03.png?raw=true)
![img04](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-04.png?raw=true)
![img05](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-05.png?raw=true)
![img06](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-06.png?raw=true)
![img07](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-07.png?raw=true)
![img08](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-08.png?raw=true)
![img09](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-09.png?raw=true)
![img10](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-10.png?raw=true)
![img11](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-11.png?raw=true)
![img12](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-12.png?raw=true)
![img13](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-13.png?raw=true)
![img14](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-14.png?raw=true)
![img15](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-15.png?raw=true)
![img16](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-16.png?raw=true)
![img17](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-17.png?raw=true)
![img18](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-18.png?raw=true)
![img19](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-19.png?raw=true)
![img20](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-20.png?raw=true)
![img21](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-21.png?raw=true)
![img22](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-22.png?raw=true)
![img23](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-23.png?raw=true)
![img24](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-24.png?raw=true)
![img25](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-25.png?raw=true)
![img26](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-26.png?raw=true)
![img27](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-27.png?raw=true)
![img28](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-28.png?raw=true)
![img29](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-29.png?raw=true)
![img30](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-30.png?raw=true)
![img31](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-31.png?raw=true)
![img32](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/efficientdet_youtube/efficientdet_youtube-32.png?raw=true)

# 4. zylo117/Yet-Another-EfficientDet-Pytorch

- Github Link : [Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
- 핵심 EfficientDet model 구현 부분 : [Yet-Another-EfficientDet-Pytorch/efficientdet/model.py](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/efficientdet/model.py) 