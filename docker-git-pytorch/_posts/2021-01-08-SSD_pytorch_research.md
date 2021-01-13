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
- 결론 
    - 내가 만약 **import package.dir1.dir2** 를 파일 맨 위에 한다면, **dir2에 있는 \_\_init\_\_.py** 이 자동으로 호출되며 안의 내용을 모두 읽고 실행한다. 
    - 만약 dir2의 \_\_init\_\_.py에 **from .dir3.dir4 import fun4** 가 있다면?
    - a.py에서 아래의 2가지 방법을 사용하면 된다. 
        - **from package.dir1 import dir2**를 한다면, a.py에서 dir2.fun4로 써놓음으로써 fun4를 사용할 수 있다!!
        - **from dir2 import fun4**라고 한다면, a.py에서 fun4를 그대로 사용할 수 있다. 
    - 원래는 a.py에서 fun4로만 생각해서, 직접 패키지를 만들고 실험을 해봤더니, 에러가 났다. 
    - debugging을 해보니까, 다음과 같은 실행이 이뤄짐을 알 수 있었다.   
        ![image](https://user-images.githubusercontent.com/46951365/104087589-39a28b00-52a4-11eb-83c7-ed0dfc47614d.png)

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
    { % highlight python linenos % }
    class VGG(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            size = cfg.INPUT.IMAGE_SIZE
            vgg_config = vgg_base[str(size)]
            extras_config = extras_base[str(size)]

            self.vgg = nn.ModuleList(add_vgg(vgg_config : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512])

    def add_vgg(cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1) # 이런식으로 해
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
        { % endhighlight % }
    ```
- 코드 설명 - 



