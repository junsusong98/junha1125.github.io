---
layout: post
title: 【Pytorch Package】SSD Pytorch 'Module' Research
description: >
    SSD package를 공부하면서 새로 익힌 내용들을 정리한다.
---  
- 아직 python에 대해서도 많은 부분을 모른다.
- 모르는게 있으면, 끝까지 탐구하는 자세는 좋다. 하지만 너무 오랜 시간은 좋지 않다. 버릴건 버리고 나중에 필요하면 그때 다시 공부하면 이해가 될 수 있다.

# 1. py파일에서 부모 directory의 py파일에 접근 
- reference 
    1. [https://myjorney.tistory.com/52](https://myjorney.tistory.com/52)
    2. [https://devlog.jwgo.kr/2019/09/17/how-to-test-package-created-by-myself/](https://devlog.jwgo.kr/2019/09/17/how-to-test-package-created-by-myself/)

- 문제점 : 다음과 같은 파일 구조에서  
    ```sh
    aa
        - a.py
        - bb
            - b.py
    ```
    b.py에서 from .. import a, from ... import a 이 잘 동작하지 않는다.
- 해결책 (위의 레퍼런스 요약)
    1. (1번째 refer) terminal에서 path를 b가 아닌 a에 두고, "$ python -m b/b.py"를 실행한다.
        - "-m" 옵션을 사용함으로써 모듈의 위치에 집중하라고 알려준다,
        - python을 실행할 때, 터미널의 pwd가 굉장히 중요하다,
    2. (**추천**-2번째 refer) setup.py를 이용한다. 
        - sys.path.append [과거 블로그 설명과 사용법](https://junha1125.github.io/docker-git-pytorch/2020-08-19-Keras-yolo3/), sys.path.insert와 같은 방법. 혹은 컴퓨터의 환경변수를 넣어주는 방법 등이 있지만 모두 비추이다.
        - 나만의 패키지를 만들 때, 가장 맨위 디렉토리에 setup.py를 만들어 준다. 양식은 [python공식문서](https://packaging.python.org/tutorials/packaging-projects/#creating-setup-py)를 참고하면 된다. 그리고 "**$python setup.py install** /or/ $ pip install ."을 해준다!!
    3. 나중에 setup.py 에 대해서 공부해보자. Google : [how work setup.py](https://stackoverflow.com/questions/1471994/what-is-setup-py)


# 2. \_\_init\_\_.py 파일의 힘 (__init__.py)
- 문제점 
    - **"아니 이게 왜 호출이 되지??? import한 적이 없는데???"** 라는 생각을 자주해서 해당경로에 \_\_init\_\_.py에 들어가보면 import를 대신 해준것을 확인할 수 있었다.
    - 그렇다면 이게 어떻게 동작하는 걸까?
    
- 결론 ⭐ 
    - 내가 만약 **import package.dir1.dir2** 를 파일 맨 위에 한다면, **dir2에 있는 \_\_init\_\_.py** 이 자동으로 호출되며 안의 내용을 모두 읽고 실행한다. 
    - 만약 dir2의 \_\_init\_\_.py에 **from .dir3.dir4 import fun4** 가 있다면?
    - a.py에서 아래의 2가지 방법을 사용하면 된다. 
        - **from package.dir1 import dir2**를 한다면, a.py에서 dir2.fun4로 써놓음으로써 fun4를 사용할 수 있다!!
        - **from dir2 import fun4**라고 한다면, a.py에서 fun4를 그대로 사용할 수 있다. 
    - 원래는 a.py에서 fun4로만 생각해서, 직접 패키지를 만들고 실험을 해봤더니, 에러가 났다. 
    - debugging을 해보니까, 다음과 같은 실행이 이뤄짐을 알 수 있었다.   
        ![image](https://user-images.githubusercontent.com/46951365/104087589-39a28b00-52a4-11eb-83c7-ed0dfc47614d.png)
    
- 궁금증2

    - ```python
        from mmdet.apis import inference_detector, init_detector, show_result_pyplot
        ```

    - 이러고 inference_detector, init_detector 함수를 그냥 사용할 수 있다. /mmdet/apis.py 파일이 있고, 저 파일안에 저 함수들이 정의되어 있다면 내가 궁금해하지도 않는다. 아래와 같은 구조를 하고 있기 때문이다.

    - ![image-20210128205145476](C:\Users\sb020\AppData\Roaming\Typora\typora-user-images\image-20210128205145476.png)

    - 물론 위의 문제점1에서 깨달았던 방법을 사용하면 아래와 같은 모듈사용이 가능했다.  

        ```python
        from mmdet import apis
        
        config = 'configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py'
        checkpoint = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
        
        model = apis.init_detector(config, checkpoint, device='cuda:0')
        ```

    - 어떻게 위의 form과 import의 실행이 가능했던 것일까??

- 결론

    - ⭐ **from**이  가르키는 마지막 directory의 \_\_init\_\_.py 또한 일단 다 읽는다! 그리고 **import**다음으로 넘어간다. 
    - 우리가 아는 아주 당연한 방법으로, **import** 다음 내용으로 **from** A.B.C **import** py_file_name/function_defined 과 같이 import를 수행해도 된다.
    - 하지만 from 가장 마지막 directory의 (위의 예시에서 C)의 \_\_init\_\_.py 안에서 한번 import된 함수를 import해서 가져와도 된다.
    - 이를 이용해서, 궁금증2의 첫 실행문을 다시 설명하자면, `from mmdet.apis` 을 통해서 apis의 \_\_init\_\_.py를 모두 읽는다. 여기서 `from .inference import (async_inference_detector, inference_detector,init_detector, show_result_pyplot)` 이 수행되므로, import에서 `import inference_detector, init_detector, show_result_pyplot`를 하는 것에 전혀 무리가 없는 것이다.



# 3. os.path 모듈
- reference : [https://devanix.tistory.com/298](https://devanix.tistory.com/298)
- os.path.abspath(path)
- os.path.basename(path)
- os.path.commonprefix(path_list)
- os.path.dirname(path) : 굳이 내가 \\로 직접 자를 필요가 없었다. 
- os.path.exists(path)
- os.path.getsize(path)
- os.path.isfile(path)
- os.path.isdir(path)
- os.path.join(path1[,path2[,...]]) : 나의 os 확인하니, 일반 string class의 + 연산자보다 이게 낫겠다. 
- os.path.normpath(path) : 현재 디렉터리(".")나 상위 디렉터리("..")와 같은 구분자를 최대한 삭제

# 4. ModuleList
- ```python
    class VGG(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            size = cfg.INPUT.IMAGE_SIZE
            vgg_config = vgg_base[str(size)]
            extras_config = extras_base[str(size)]

            self.vgg = nn.ModuleList(add_vgg(vgg_config : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]))

    def add_vgg(cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1) 
                if batch_norm:ㅍ
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        layers += [pool5, conv6,
                nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
        return layers
    ```
- 코드 설명 
    - ModuleList()에는 input으로 python-list가 들어가고 list-component는 nn.layer 이다. 
    - conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1) 이 문장.
        - conv2d은 계속 새로운 nn.Conv2d 객체로 바끤다.
        - 아래의 코드를 참고해보자.
        - ```python
            >>> a = (1,2)
            >>> id(a)
            2289005091144
            >>> id((1,2))
            2289005102280
            >>> a = (1,2)
            >>> id(a)
            2289005092296
            ```
        - 이처럼 같은 클래스 tuple이지만, 객체 (1,2)는 다시 만들어지고 그것을 a가 가리킨다.



# 5. \_\_init\_\_() defined in Class
- Class의 __init__함수는 객체 선언시 처음부터 끝까지 읽혀진다.
- 아래의 코드에서도 self.set_c()까지 처리되는 것을 확인할 수 있다.
- ```python
    class a():
    def __init__(self, a,b):
        self.a = a
        self.b = b
        self.c = 1
        self.set_c()

    def set_c(self):
        self.c = self.a + self.b

    obj = a(4,3)
    print(obj.c) 
    >> 7
    ```
- ```python
    class VGG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        size = cfg.INPUT.IMAGE_SIZE
        vgg_config = vgg_base[str(size)]
        extras_config = extras_base[str(size)]

        self.vgg = nn.ModuleList(add_vgg(vgg_config))
        self.extras = nn.ModuleList(add_extras(extras_config, i=1024, size=size))
        self.l2_norm = L2Norm(512, scale=20)
        self.reset_parameters()
    ```
- 따라서 위 코드의 reset_parameters도 반드시 실행된다.

# 6. torch.tensor.function (permute, contiguous)
- ```python
    cls_logits = []
    bbox_pred = []
    for feature, cls_header, reg_header in zip(features, self.cls_headers, self.reg_headers):
        cls_logits.append(cls_header(feature).permute(0, 2, 3, 1).contiguous())
        bbox_pred.append(reg_header(feature).permute(0, 2, 3, 1).contiguous())
    ```
- 위의 cls_header, reg_header 는 nn.Conv2d 이다. 
    - cls_header(feature) : nn.Conv2d를 통과하고 나온 feature(type = tensor) 이다.  
    - [torch.Tensor.permute](https://pytorch.org/docs/stable/tensors.html?highlight=permute#torch.Tensor.permute) 를 사용해서 tenor 연산 이뤄지는 중
    - [torch.Tensor.contiguous](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.contiguous)
- torch document에 적혀 있는 view는 tensor의 shape, size를 의미한다. view라는 단어를 더 자주쓰지 하나의 명사로 알아두기
    
    - [torch.tensor.view](https://pytorch.org/docs/stable/tensors.html?highlight=permute#torch.Tensor.permute)

# 7. os.environ(\["variable"\])
- 터미널 내부 환경 변수를 읽어오는 모듈이다.
- $ export COCO_ROOT="/path/to/coco_root"  와 같이 환경변수를 설정해 두면, python내부 os.environ 함수가 이 환경변수를 읽오와 변수에 저장해둔다.
- <img src='https://user-images.githubusercontent.com/46951365/104723082-0bc0b900-5772-11eb-99cd-655fd6631f2a.png' alt='drawing' width='300'/>

# 8. package 다운 후 코드 수정하기
- 궁금증 : 
    - $python setup.py install / $ pip install .  
    - 이 명령어를 이용해서, package의 setup을 마친 상태라고 치자. 
    - setup을 하고 난 후, package의 코드를 바꾸면 다시 setup을 실행해줘야 하는가?
    - 아니면 setup은 처음에만 하고, 코드 수정 후 그냥 사용하기만 하며 되는가?

- 해답 : 
    - 정답은 **'처음에만 setup 해주면 된다.'** 이다
    - 아래와 같은 세팅의 실험에서, 내가 추가해준 코드가 아주 잘 실행되는 것을 확인할 수 있었다.
    - <img src="https://user-images.githubusercontent.com/46951365/105042656-d4bb1200-5aa7-11eb-9dc0-bc8eb2cc1a83.png" alt="image" style="zoom:67%;" />
