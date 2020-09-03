---
layout: post
title: 【Keras】 GPU 학습 유의사항/ 라쿤 데이터 학습시키기 - Keras기반 Yolo3 
description: >
    GPU Object Detection시 유의 사항을 간단히 알아보고, 이전 게시물에서 공부했던 Keras 기반 Yolo3를 이용해 데이터 학습을 시켜보자.
---

# 1. GPU 학습 시킬 떄 주의 사항
1. 대량의 이미지 학습시 메모리 문제
    - 알다시피 개별 이미지가 Tensor, narray로 변환해서 NeuralNet에 집어넣을떄 이미지를 Batch로 묶어 처리한다. 
    - 이 Batch가 그래픽 카드 메모리 사용량의 많은 부분을 좌우한다.  
    (1000장 - 10장 배치 * 100번 = 1 에폭)
    
2. Genrator와 Batch
    - 특히 fit_generator를 확인해보면, Python gererator를 사용한다. gerator, next, yield에 대해서 알아보자. [코딩도장 generator 설명](https://dojang.io/mod/page/view.php?id=2412)
    - 아래의 이미지를 먼저 다 읽어보기
    - 배치 사이즈를 크게 늘린다고 학습이나 추론이 빨라지지 않는다. 100장의 이미지를 한방에 끌어오는것도 힘들지만, 그 이미지를 네트워크에 넣는다 하더라도 어딘가에서 병목이 걸릴건 당연하다. 따라서 배치 사이즈를 어떻게 설정하지는 CPU core, GPU 성능 등을 파악해서 균형있게 맞춰야 한다.(Heuristic 하게 Turning)
    - 한쪽은 열심히 일하고, 한쪽은 기다리는 현상이 일어나지 않게끔. 이렇게 양쪽이 일을 쉬지 않고 균형있게 배치하는 것이 필요하다. 
    - **이러한 Batch를 Keras에서는 (fit_generator 함수가 사용하는) DataGenerator, flow_from_drectory 가 해준다. 알다시피 Torch는 TensorDataset, DataLoader가 해준다.**
    - 이런 generator를 사용하기 위해서 Keras와 같은 경우에 다음과 같은 코드를 사용한다. 
    
        ```python
            from keras.preprocessing.image import ImageDataGenerator

            train_datagen = ImageDataGerator(rescale=1/255) # 이미지 정규화
            train_gernator = train_datagen.flow_from_director('학습용 이미지 Directory', targer_size = (240,240), batch_size-10, class_mode='categoricl') 
            valid_datagen = ImageDataGerator(rescale=1/255) 
            valid_gernator = valid_datagen.flow_from_director('검증용 이미지 Directory', targer_size = (240,240), batch_size-10, class_mode='categoricl')

            # 모델을 구성하고
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(240,240,3)))
            model.add ...
            ...
            ...
            model.add(Dense(3, activation='softmax'))
            model.compile(loss='categorial_crossentropy', optimizer='adam', metrixs=['accuracy'])

            # 학습을 시킨다.
            model.fit_generator(train_generator, steps_per_epoch=100, epochs=10, validation_data=valid_generator, validation_steps=2)
        ```
    - 이렇게 fit_generator을 돌리면 data pipeline이 만들어 진다. train_generator가 디렉토리에 가서 배치만큼의 파일을 읽어와서 yeild를 하여 모델에 데이터를 넣는다. 


<p align="center"><img src='https://user-images.githubusercontent.com/46951365/92118150-0bfa9080-ee31-11ea-8398-6de32d892007.png' alt='drawing' width='500'/></p>


# 2. qqwweee/Keras-Yolo3로 Training 하기 위한 분석

- 지금까지는 1. Pretrained된 Weight값을 가져와서 Inference를 수행하거나(Tensorflow(SSD, Yolo), Keras-yolo3) 2. OpenCV DNN 모듈을 사용하거나 해왔다.
- 참고 : 1개의 객체만 검출한다면 100장이어도 충분하다. 근데 Class가 여러개면 수 많은 이미지 데이터가 필요하다. 
- \<weight 가중치 파일 형식 정리>
    - **keras weight(케라스 가중치) 파일은 파일 형식이 .h5**파일이다. 
    - Tensorflow에서 Inference를 위한 파일 형식은 .pb 파일이다. .ckpt파일에는 학습에 대한check point에 대한 정보를 담고 있다. 
    - PyTorch에서는 .pt 파일이 torch.save(model.state_dict()를 이용해서 저장한 weight파일이다. 
- qqwweee/keras-yolo3에 적힌 Training 방법 정리
    1. voc(xml 파일사용)을 annotation파일로 바꾸는 convert_annotation.py 파일을 살펴보면 최종적으로 '%s_%s.txt'%(year, image_set) 파일을 만드는 것을 확인할 수 있다. 
    2. Readme의 내용을 따르면,  
    각 한줄 : image_file_path box1 box2 ... boxN  
    boxK의 format : x_min,y_min,x_max,y_max,class_id (no space) 이다.   
        ```sh
        path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
        path/to/img2.jpg 120,300,250,600,2
        ...
        ```  
    3. 일단 pretrained된 weight를 가져와서 학습시키는 것이 좋으니, conver.py를 실행해서 darknet weight를 keras weight로 바꿔준다.   
    4. train.py를 사용해서 학습을 한다. 내부 코드 적극 활용할 것.
    5. train.py에 들어가야하는 Option을 설정하는 방법은 원래의 default option이 무엇인지 확인하면 된다. 

<p align="center"><img src='https://user-images.githubusercontent.com/46951365/92137843-a31f1280-ee48-11ea-9094-84e267048ef3.png' alt='drawing' width='300'/></p>

- train.py을 적극적으로 이용할 계획이다. 아래와 같은 방법으로. 함수는 아래와 같이 import를 해서 사용할 것이고, main함수의 내용은 소스 코드 그대로 복사해 가져와서 사용할 예정이다.   
    ```python
        from train import get_classes, get_anchors
        from train import create_model, data_generator 
    ```


# 3. 라쿤 데이터셋으로 학습시키고 Inference하기

- [좋은 데이터 세트 모음](https://github.com/experiencor/keras-yolo3) : [라쿤데이터셋](https://github.com/experiencor/raccoon_dataset)