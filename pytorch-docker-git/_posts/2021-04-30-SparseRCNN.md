---
layout: post
title: 【Pytorch】Sparse-RCNN(detectron2) teardown reports
---

Sparse-RCNN teardown reports 



**최종목표!!** 

내가 생각한 아이디어 완전히 조져버리기!



# 1. docker build & run

```sh
$ sudo docker run -d -it      \
    --gpus all         \
    --restart always     \
    -p 8000:8080         \
    --name "Sparse-RCNN"          \
    --shm-size 8gb      \
    -v /home/junha/docker:/workspace  \
    -v /ssdB:/dataset   \
    sb020518/detectron2:4.29
```

`SparseRCNN`부분의 코드를 수정하면, 수정된 코드가 반영되고 싶은데, `Detecrton2-repo`의 코드가 수정되어야 수정된 코드가 반영된다 따라서 아래의 작업을 

```sh
$ pip uninstall detectron2
$ git clone https://github.com/PeizeSun/SparseR-CNN.git
$ cd SparseR-CNN
$ python setup.py build develop
```





# 2. Sparse R-CNN github

코드 링크 : [https://github.com/PeizeSun/SparseR-CNN](https://github.com/PeizeSun/SparseR-CNN)

Detectron 기초 공부 : [detectron2 teardown reports](https://junha1125.github.io/blog/pytorch-docker-git/2021-04-29-detectron2/)

**Dataset Setting**

```sh
mkdir -p datasets/coco
ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2017 datasets/coco/train2017
ln -s /path_to_coco_dataset/val2017 datasets/coco/val2017
```

**Train SparseR-CNN**

```sh
python projects/SparseRCNN/train_net.py --num-gpus 2 \
    --config-file projects/SparseRCNN/configs/sparsercnn.res50.100pro.3x.yaml
```

**Evaluate SparseR-CNN**

```sh
python projects/SparseRCNN/train_net.py --num-gpus 2 \
    --config-file projects/SparseRCNN/configs/sparsercnn.res50.100pro.3x.yaml \
    --eval-only MODEL.WEIGHTS checkpoints/r50_100pro_3x_model.pth
```

**Visualize SparseR-CNN**

```sh
# 나의 설정에 맞춤
$ python demo/demo.py\
	--config-file projects/SparseRCNN/configs/sparsercnn.res50.100pro.3x.yaml\
	--input demo/dog.jpg\ 
	--output demo/.\
	--confidence-threshold 0.4\
    --opts MODEL.WEIGHTS checkpoints/r50_100pro_3x_model.pth
```



# 3. error 발생과 해결

```sh
python projects/SparseRCNN/train_net.py --num-gpus 2 \
    --config-file projects/SparseRCNN/configs/sparsercnn.res50.100pro.3x.yaml \
    --eval-only MODEL.WEIGHTS checkpoints/r50_100pro_3x_model.pth
```

위 명령어를 돌리면 서버가 잠깐 끊겨서 접속이 끊기는 문제가 발생했다. 아래에 내가 했던 시도를 정리해 놓는다.

1. (안정적으로 돌리기) vscode container window에서 F5를 눌러서 run하지 말고, terminal에서 직접 위에 커멘드를 실행한다. 혹은 VScode ssh window에서 run해본다. 그래도 안되면 terminal-ssh 연결 상태에서 직접 run 해본다. 
   - container terminal : 같은 문제 발생
   - vscode ssh terminal : 같은 문제 발생
   - Iterms 에서 ssh연결 : 서버 연결이 끊기지는 않는 것처럼 보이지만, 터미널에 아무 동작이 일어나지 않는다. 하지만! `Image_read` print되는 속도가 매우 느리다. 마치 쓰레드끼리 얽히고 섥혀서 각자의 process가 끝나기만을 기다리며.. 아무것도 안하고 있는 것처럼.
2. detectron2에서도 같은 문제 발생! container를 바꿔서 detectron2에서 FasterRCNN을 evaluation해보았다. 놀랍게도 똑같은 문제가 발생했다. 이것으로 SparseRCNN 개발자 컴퓨터환경과 달라서 발생하는 문제가 아닌, Detectron2 개발자들 모두의 환경에서 잘 되는게 나에게 문제가 발생하는 것이므로, 그냥 내 컴퓨터에서 문제가 있는 것 같다. 
   - detectron2 run 명령어 `tools/train_net.py --num-gpus 2  --config-file configs/COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml  --eval-only MODEL.WEIGHTS checkpoints/model_final_3e0943.pkl`
   - -v 옵션으로 mount한 것이, HDD라서 생각보다 dataloading이 느려서 그런가? 같은 path 로 dataset을 옮겨 놓자! 같은 path ssd로 옮겼지만, 결과는 똑같다.
3. (이상한 방법) 디버깅을 많이 하다보니, 문제의 장소를 찾았다. `detectron2/evaluation/evaluation.py inference_on_dataset 맴버함수` 에서 `output=model(inputs); print("stop temporary"); time.sleep(0.1)` 처리를 해주었다. 이렇게 하니 서버 접속은 잘 유지되었다. 
   - detectron2는 쓰레드를 사용해서 dataloding을 한다. 알다 싶이, 쓰레드의 동작은 아무도 예측할 수 없다. 무엇이 이유없이 먼저 실행되고 무엇은 이유없이 늦게 실행된다. 실행 순서도 뒤죽박죽이다. 
   - 그래서 dataloading을 하는 `detectron2/data/detectron_utils.py read_image 맴버함수` 에  `print(image read)` 명령어를 넣어서 확인해본 결과 `print(stop temporary) ; print(read)` 가 뒤죽박죽이다. 
   - 그리고 쓰레드 하나가 정확하게 한장의 이미지를 load하는게 아닌 것 같다. 쓰레드 3개가 차곡차곡 이미지 한장의 데이터를 GPU에 넣어주는 것과 같은 모습이 보이기도 한다. 이렇게 생각한 이유는 batch가 4이면, `print(read)`가 딱 4번만 실행되어야 하는데 4번이상 실행되고 그러더라. 
4. (위에 time.sleep를 사용한다는 전재 하에), (batch size & worker size) worker size=32 IMS_PER_BATCH(=batch size per GPU)=16으로 해도 잘 돌아간다. 무조건 낮추는게 답은 아니었다. 낮춰서 받아야할 데이터를 못받아서 문제가 생기는 경우도 발생하는 것 같다. 
   - [[코어 검색 방법](https://brownbears.tistory.com/334)] CPU 코어 8개, 각 코어당 물리코어 8개, 총 48개 물리코어. [[Thread 갯수 확인법=최대 프로세스 개수](http://blog.cloudsys.co.kr/linux-threads-max-count-sysctl/)] 384763
   - 따라서 내가 worker size를 엄청 작게 할 필요가 없다. 디버깅을 해보니까 worker_size는 `Number of data loading threads`이라고 한다. 따라서 batch를 크게 설정해도 될 듯 하다. 
   - Worker64 batch36 해도 된다. 지금 말이 되나? GPU 사용량은 그냥 30퍼 정도이다. worker와 batch를 이정도 늘려도 30퍼 수준이다. 
   - Worker2 batch2로 해도 똑같이. GPU 사용량은 그냥 30퍼 정도이다. worker와 batch를 이정도 늘려도 30퍼 수준이다. 
   - 이 야매 방법을 사용했을때 왜 위와 같은 문제가 발생하는지 모르겠다. 
5. (Time.sleep을 적용하지 않았을 때.) worker 2 batch 2로 하면 서버 접속이 끊긴다. 
   - 내가 최종적으로 내린 결론은. 전력 부족이다.
   - worker, batch 를 크게하고 sleep을 적용하면 어떤 size더라도 GPU사용량 30퍼 만을 사용한다. 
   - worker 2 batch 2를 하고 sleep을 적용하면, 서버 접속이 끊기기 전에 GPU의 사용량이 70~80퍼 이다. 
   - 따라서 나의 결론! 전력 부족 가능성이 가장 크다.
   - 현재 1080Ti 2개 Power 700w.. 이다. 본체의 Power를 1000w 짜리로 업그레이드 하니 모든 문제 해결
   - **결론: 본체 파워 용량 부족**



# 4. Evaluation 처리 순서 (for visualization)

1. /projects/**SparseRCNN**/train_net.py
   - res = Trainer.test(cfg, model)
2. /**detectotron2**/engine/defaults.py > DefaultTrainer > test
   - results_i = inference_on_dataset(model, data_loader, evaluator)
3. /**detectron2**/evaluation/evaluator.py > inference_on_dataset
   - results = evaluator.evaluate()
   - evaluator의 맴버변수 `self._predictions`에 모든 이미지에 대한 모델 추론 결과 저장되어 있음
4. /**detectron2**/evaluation/coco_evaluation.py > COCOEvaluator > _eval_predictions
   - coco_eval = _evaluate_predictions_on_coco( coco_json , 모델 예측결과 , task='bbox'(object detection))
5. /**detectron2**/evaluation/coco_evaluation.py > _evaluate_predictions_on_coco()
   - coco_dt = coco_gt.loadRes(coco_results)
6. /pip-packages/**pycocotools**/coco.py > COCO class > loadRes 맴버함수
   - 디버깅을 위해서 launch.json 파일에 `"justMyCode" : false` config 추가
   - 오늘은 코드를 계속봐서 눈에 안들어 온다. 남은 시간은 논문 읽고 다음에 코드 더 보자. 
   - 혹은 DETR의 코드를 이용해도 좋을 것 같다. ㅇ





# 5. Code 수정

1. Config 에 새로운 변수 추가하는 방법
   - from sparsercnn import SparseRCNNDatasetMapper, **add_sparsercnn_config**
   - **add_sparsercnn_config**(cfg)
   - 해당 파일을 보면, `Add config for SparseRCNN.` 을 위한 방법이 나와있다.
2. 원래 패키지에서 아래와 같이 수정시 AP가 1프로 밖에 안떨어짐
   `proposal_boxes = torch.stack([torch.tensor([0.5, 0.5, 1, 1]) for _ in range(self.num_proposals)]).to(self.device)`
3. 
