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
python projects/SparseRCNN/train_net.py --num-gpus 8 \
    --config-file projects/SparseRCNN/configs/sparsercnn.res50.100pro.3x.yaml
```

**Evaluate SparseR-CNN**

```sh
python projects/SparseRCNN/train_net.py --num-gpus 8 \
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



**Config**

```

```







