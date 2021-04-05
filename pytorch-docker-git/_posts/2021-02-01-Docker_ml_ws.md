---

layout: post
title: 【docker】container setting using ML-workspace 
---

- Github Link - [ml-tooling/ml-workspace](https://github.com/ml-tooling/ml-workspace)
- 과거에 공부했던 [docker 기본 정리 Post 1](https://junha1125.github.io/blog/pytorch-docker-git/2020-02-18-tstory_7/)
- 과거에 공부했던 [docker 기본 정리 Post 2](https://junha1125.github.io/blog/pytorch-docker-git/2020-02-21-tstory_8/)



## 1. [install docker](https://docs.docker.com/engine/install/ubuntu/) 



## 2. 필수 명령어 

- ```sh
  $ sudo docker image list
  $ sudo docker image rm <img_name>
  $ sudo docker container list -a 
  $ sudo docker container rm <container-ID>
  $ sudo docker container stop <container-ID>
  $ sudo docker container start <container-ID>
  ```



## 3. How to RUN ml-workspace

- ```sh
  docker run -d \
      -p 8080:8080 \  # port 설정 
      --name "ml-workspace" \
      -v "${PWD}:/workspace" \
      --env AUTHENTICATE_VIA_JUPYTER="mytoken" \
      --shm-size 512m \
      --restart always \
      mltooling/ml-workspace:0.12.1
  ```

- -p : [port 설정 설명](https://www.youtube.com/watch?v=pMY_wPih7R0&list=PLEOnZ6GeucBVj0V5JFQx_6XBbZrrynzMh&index=3), -p : port설정

- -v : 나같은 경우 `"$pwd"/docker_ws:/workspace`

  - [bind mount](https://docs.docker.com/storage/bind-mounts/)
  - 나의 컴퓨터 terminal의 현재 **"$pwd"**/docker_ws  (나같은 경우 **/home/junha**/docker_ws)
  - container의 new ubuntu root에 /workspace라는 폴더 생성 후

- -env AUTHENTICATE_VIA_JUPYTER="내가 설정하고 싶은 비밀번호"

- -d : background container start



## 4. Docker container와 VScode 연동하기

1. 첫번째 VScode 사용하는 방법
   - [ml-workspace에서 알려주는 vscode](https://github.com/ml-tooling/ml-workspace#visual-studio-code)
   - vscode 서버를 사용한 실행.
   - 원하는 폴더를 체크박스_체크 를 해주면, vscode옵션을 선택할 수 있다. 
   - 이렇게 하면 EXTENSIONS 를 나의 컴퓨터 그대로 사용하기는 어려우므로 VScode docker 연동법을 공부해보자.
2. 컴퓨터 VScode의 docker SSH 를 사용하서 연결하기.
   - Remote-container EXTENSION을 설치해준다. 
   - ctrl+shift+p -> Remote-Containers: Attach to Running Container
     - 자동으로 현재 container 목록이 보인다. 원하는거 고르면 remote vscode 새창 뜬다
     - 아주 잘 됨.
   - 굳이 IP설정을 해주지 않더라도 잘 연결된다. ([GCP에서의 SSH 설정하기](https://junha1125.github.io/blog/ubuntu-python-algorithm/2020-09-19-SSHvscode/) 이것과는 달리..)
   - 만약 문제가 생기면 config 파일 여는 방법
   - ctrl+shift+p -> Remote-Containers: Open Attached Container Configuration File
   



## 5. detectron2 using ml-workspce 

1. ```sh
   import torch, torchvision
    print(torch.__version__, torch.cuda.is_available())

    import detectron2
    from detectron2.utils.logger import setup_logger
    setup_logger()

    # import some common libraries
    import numpy as np
    import os, json, cv2, random

    # import some common detectron2 utilities
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog, DatasetCatalog

    im = cv2.imread("./input.jpg")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.DEVICE = "cpu" # 필수!!!!
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
   ```
   
2. ```sh
   $ python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.7/index.html
   $ pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
   ```
   
3. 1번을 그대로 실행하니 에러발생. 2번의 과정을 통해서 에러 해결.

4. 디버깅이 안된다. Step into로 들어가면, 내가 보고 있는 py파일 뿐만 아니라, Import하는 함수 내부로 들어가서 어떤 파일과 함수가 실행되는지 보고 싶은데 안된다. docker container 내부, ubuntu의 env 속, python 모듈은 디버깅으로 들어갈 수 없는 건가??
   
5. 에러내용 - DefaultPredictor가 어떤 흐름으로 실행되는지 대충 훔처 볼 수 있다.    

    ```sh
    AssertionError:
    Torch not compiled with CUDA enabled
    # /opt/conda/bin/python /workspace/test.py
    1.7.1 False
    ** fvcore version of PathManager will be deprecated soon. **
    ** Please migrate to the version in iopath repo. **
    https://github.com/facebookresearch/iopath 
    
    ** fvcore version of PathManager will be deprecated soon. **
    ** Please migrate to the version in iopath repo. **
    https://github.com/facebookresearch/iopath 
    
    model_final_f10217.pkl: 178MB [01:03, 2.79MB/s]                                                                                                                                                        
    # MODEL.DEVICE = 'cpu' 
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    test.py 29 <module>
    outputs = predictor(im)
    
    defaults.py 223 __call__
    predictions = self.model([inputs])[0]
    
    module.py 727 _call_impl
    result = self.forward(*input, **kwargs)
    
    rcnn.py 149 forward
    return self.inference(batched_inputs)
    
    rcnn.py 202 inference
    proposals, _ = self.proposal_generator(images, features, None)
    
    module.py 727 _call_impl
    result = self.forward(*input, **kwargs)
    
    rpn.py 448 forward
    proposals = self.predict_proposals(
    
    rpn.py 474 predict_proposals
    return find_top_rpn_proposals(
    
    proposal_utils.py 104 find_top_rpn_proposals
    keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
    
    nms.py 21 batched_nms
    return box_ops.batched_nms(boxes.float(), scores, idxs, iou_threshold)
    
    _trace.py 1100 wrapper
    return fn(*args, **kwargs)
    
    boxes.py 88 batched_nms
    keep = nms(boxes_for_nms, scores, iou_threshold)
    
    boxes.py 41 nms
    _assert_has_ops()
    
    extension.py 62 _assert_has_ops
    raise RuntimeError(
    
    RuntimeError:
    Couldn't load custom C++ ops. This can happen if your PyTorch and torchvision versions are incompatible, or if you had errors while compiling torchvision from source. For further information on the compatible versions, check https://github.com/pytorch/vision#installation for the compatibility matrix. Please check your PyTorch version with torch.__version__ and your torchvision version with torchvision.__version__ and verify if they are compatible, and if not please reinstall torchvision so that it matches your PyTorch install.
    ```

    



## 6. Debug Docker Containers 

1. 명학이의 도움으로 다음의 과정을 진행해 보았다. -> 안됨

   1. ctrl + shift + p -> setting -> `> debug : launch json` 검색-> Python file 선택
   2. 명학이가 보내준 사진대로 설정. 안됨.

2. VS code Docker debuging : 지금 당장 나에게 맞는 방법은 아니다.

   1. [https://code.visualstudio.com/docs/containers/debug-common](https://code.visualstudio.com/docs/containers/debug-common)
   2. [https://code.visualstudio.com/docs/containers/debug-python](https://code.visualstudio.com/docs/containers/debug-python)

3. Youtube로 공부해보기 -> 나에게 맞는 정보 없음

   - Link : [How to debug Docker containers! (Python + VSCode)](https://www.youtube.com/watch?v=qCCj7qy72Bg) 

     1. Use **docker exec**
        - ```sh
          $ docker run -d \
              -p 8080:8080 \  # port 설정 
              --name "ml-workspace" \
              -v "${PWD}:/workspace" \
              --env AUTHENTICATE_VIA_JUPYTER="mytoken" \
              --shm-size 512m \
              --restart always \
              --entrypoint=bash
              mltooling/ml-workspace:0.12.1
          ```

        - ```sh
          $ docker exec -it <docker-container-name or ID> bash
          # new shell start 
          $ python
          >> 여기서 module 디버깅
          ```

     2. Use **Docker cp** 

        - print해서 나올 파일을 text.txt파일에 저장되도록 만든다.

        - ```python
          with open("time.txt",'w+') as f:
          	f.write(str(<출력하고 싶은 변수>))
          ```

        - **docker cp command** 사용하기

        - ```sh
          $ docker cp <docker-container-name or ID>:/absolut/path/filename ./my_ubuntu/terminal/relative/path
          ```

4. 직접 문제 해결
  
   - 나의 문제는 docker remote container가 실행되고 있는 상태(아래 이미지)에서 step-into dugging이 안되는 것이다. 
   
   - ![image](https://user-images.githubusercontent.com/46951365/107231156-4bbe4780-6a63-11eb-8cc2-2b1d330d7f83.png)
   
   - Google 검색 : "vscode python debug step into not working"
   
   - 참고한 사이트 : [stack overflow - Python debugging - Step into](https://stackoverflow.com/questions/53594900/visual-studio-code-python-debugging-step-into-the-code-of-external-functions)
   
   - 우분투에서 ctrl + shift + p -> `> debug : launch json` 검색-> Python file 선택 이 파일 내부의 내용을 아래 처럼 수정한다.    
     
     ```
     {
          // Use IntelliSense to learn about possible attributes.
          // Hover to view descriptions of existing attributes.
          // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
          "version": "0.2.0",
          "configurations": [
              {
                  "name": "Python: Current File",
                  "type": "python",
                  "request": "launch",
                  "program": "${file}",
                  "console": "integratedTerminal",
                  "debugOptions" : ["DebugStdLib"],
                  "justMyCode": false        
                }
          ]
      }
     ```
     
   - 하지만!! SSH server's Docker container에서는 debug json 검색해도 찾을 수가 없었다. 더 쉬운 다른 방법은 이것을 클릭하면 된다!    
     ![image](https://user-images.githubusercontent.com/46951365/113506662-22142f00-9581-11eb-876e-8a2e974a0205.png)
     
   - ```json
     
     {
          // Use IntelliSense to learn about possible attributes.
          // Hover to view descriptions of existing attributes.
          // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
          "version": "0.2.0",
          "configurations": [
              {
                  "name": "Python: Current File",
                  "type": "python",
                  "request": "launch",
                  "program": "${file}",
                  "console": "integratedTerminal",
                  "debugOptions" : ["DebugStdLib"],
                  "justMyCode": false        
                }
          ]
      }
     ```
     
   - step into 된다!! 
   
   - ![image](https://user-images.githubusercontent.com/46951365/107231731-fafb1e80-6a63-11eb-8e70-e37fa466302e.png)
   
   - ![image](https://user-images.githubusercontent.com/46951365/107231859-241baf00-6a64-11eb-8bf8-b412e1652b57.png)
   
   - 2번째 에러 : Module cv2 has no member
   
   - 해결 방법 : [https://12340zszs.tistory.com/67](https://12340zszs.tistory.com/67)
   
   - ![image](https://user-images.githubusercontent.com/46951365/107232227-84125580-6a64-11eb-9d18-165c6edfae3f.png)















