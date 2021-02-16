---
layout: post
title: 【Attention】Attention Mechanism Overveiw
---

Attention and Transformer is salient mechanism in deep learning nowadays. I remain my note which I learn from reference.



# 0. reference

1. [a-brief-overview-of-attention-mechanism](https://medium.com/syncedreview/a-brief-overview-of-attention-mechanism-13c578ba9129)
2. [Attention mechanism with schematic](https://medium.com/heuritech/attention-mechanism-5aba9a2d4727)
3. [the development of the attention mechanism](https://buomsoo-kim.github.io/attention/2020/01/01/Attention-mechanism-1.md/)
4. [How Transformers work in deep learning and NLP: an intuitive introduction](https://theaisummer.com/transformer/)





# 1. a-brief-overview-of-attention-mechanism

1. What is Attention?
    - hundreds of words -압축-> several words : 정보 손실 발생.
    - 이 문제를 해소해주는 방법이 attention. : Attention allow translator focus on local or global feature(문장에 대한 zoom in or out의 역할을 해준다.)
    - 복잡하거나 어령루거 하나도 없다. 그냥 Vector 이자 Parameters 이다. we could plug it anywhere!
2. Why Attention?
    - **Vanilla RNN** : impractical. the length of input != output. Gradient Vanishing/Exploding가 자주 일어난다. when sentences are long (more 4 words).
        - <img src='https://user-images.githubusercontent.com/46951365/104831950-b18a3a00-58d0-11eb-859a-4fabc653a69c.png' alt='drawing' width='400' style="zoom: 150%;" />
        - 한 단어씩 차례로 들어가 enbedding 작업을 거쳐서 encoder block에 저장된다, 각 block에는 각 hidden vector가 존재한다. 4번째 단어가 들어가서 마지막 encoder vertor(hidden vector inside)가 만들어 진다. 그것으로 Decoder가 generate words sequentially.
        - Issue : one hidden state really enough?
3. How does attention work?
    - 댓글에, 쓰레기 개시물이란다. 이거로 이해하지 말자. 
    - 아래의 게시물로 쉽게 contect vector, attetion vector에 대해서 이해할 수 있다.





# 2. **Attention mechanism** (awesome)

1. visual attention
    - many animals focus on specific parts
    - we should select the most pertinent piece of information, rather than using all available information. (전체 이미지 No, 이미지의 일부 Yes)
    - 사용 영역 : speech recognition, translation, and visual identification of objects
2. Attention for Image Captioning
    - Classic 이미지 캠셔닝은 아래의 방법을 사용한다. Image를 CNN을 통해서 encoder해서 feature를 얻는데 그게 Hidden state h 가 된다. 이 h를 가지고, 맨 위부터 LSTM를 통해서 첫번째 단어와 h1을 얻고, h와 h2를 LSTM에 넣어서 2번째 단어를 얻는다.
    - <img src='https://user-images.githubusercontent.com/46951365/104832643-de8d1b80-58d5-11eb-8c18-38f6e15aaee9.png' alt='drawing' width='300' style="zoom:150%;" />
    - 나오는 각각의 단어는 이미지의 한 부분만 본다.따라서 이미지 전체를 압축한 h를 /일정 부분만을 봐야하는 LSTM에/ 넣는 것은 비효율적이다. 이래서 attention mechanism이 필요하다.
    - attention mechanism
        - 이미지를 n 장으로 나눈다. 
        - CNN에 N장을 넣어서 h1_encodering, h2_encodering ... hn_encodering 를 얻는다.
        - 각각의 hk_encodering가 LSTM에 들어갈 때, hk_encodering가 집중하는 영역이 모두 다르다.
        - 그래서 decoder 각각이 바라보는 part가 모두 다르다.
        - 아래의 그림에서 attention model이 무엇인지 아래에서 공부해 보자.
        - <img src='https://user-images.githubusercontent.com/46951365/104832883-fd8cad00-58d7-11eb-9cf1-50dd08a0d801.png' alt='drawing' width='300' style="zoom:150%;" />
3. What is an attention model?
    - <img src='https://user-images.githubusercontent.com/46951365/104832942-991e1d80-58d8-11eb-8222-a10165e727e3.png' alt='drawing' width='600'/>
    - attention Model에는 n개의 input이 들어간다. 위의 예시에서 h가 y가 되겠다.
    - 출력되는 z는 모든 y의 summary 이자, 옆으로 들어가는 c와 linked된 information이다.
    - 각 변수의 의미
        - C : context,beginning.  
        - Yi : the representations of the parts of the image.  
        - Z : 다음 단어 예측을 위한 Image filter 값.
    - attention model 분석하기
        1. tanh를 통해서 m1,...mn이 만들어 진다. mi는 다른 yj와는 관계없이 생성된다는 것이 remark point이다.   
            ![img](https://miro.medium.com/max/191/0*wK-64Sr5jlKWCfjB)
        2. softmax를 사용해서 각 yi를 얼마나 비중있게 봐야 하는가?에 대한 s1,s2...sn값이 나온다. argmax가 hard! softmax를 soft라고 보면 이해하기 쉽다.   
            <img src='https://user-images.githubusercontent.com/46951365/104833528-b7861800-58dc-11eb-96ee-ed3991b41ab9.png' alt='drawing' width='200'/>
        3. Z는 s1,s2...sn와 y1,y2...yn의 the weighted arithmetic mean.  
            ![img](https://miro.medium.com/max/79/0*minF1zg5i3qHABPw)
    - modification
        - **tanh to dot product** : any other network, arithmetic(Ex.a **dot product**) 으로 수정될 수 있다. a dot product 는 두 백터의 연관성을 찾는 것이니, 좀 더 이해하기 쉽다.
        - **hard attention** 
            - 지금까지 본 것은 “Soft attention (differentiable deterministic mechanism)” 이다. hard attention은 a stochastic process이다. 확률값은 si값을 사용해서 랜덤으로 yi를 선택한다. (the gradient by Monte Carlo sampling)      
            - <img src='https://user-images.githubusercontent.com/46951365/104833686-d89b3880-58dd-11eb-8844-af1f4e19026b.png' alt='drawing' width='350' style="zoom:150%;" />
            - 하지만 gradient 계산이 직관적인 Soft Attention를 대부분 사용한다.
    - 결론적으로 위와 같은 그림으로 표현될 수 있다. **LSTM은 i번째 단어를 예측**하고 **다음에 집중해야하는 영역에 대한 정보를 담은 h_i+1**을 return한다. 
4. Learning to Align in Machine Translation (언어 모델에서도 사용해보자)
    - Image와 다른 점은 attention model에 들어가는 y1,y2 ... yi 값은, 문자열이 LSTM을 거쳐나오는 연속적인 hidden layer의 값이라는 것이다.
    - <img src='https://user-images.githubusercontent.com/46951365/104833840-1ba9db80-58df-11eb-8e1a-2d9050ffc3b1.png' alt='drawing' width='350' style="zoom:150%;" />
    - Attention model을 들여다 보면, 신기하게도 하나의 input당 하나의 output으로 matcing된다. 이것은 translation 과제에서 장점이자, 단점이 될 수 있다. 





# 3. the development of the attention mechanism

<img src='https://user-images.githubusercontent.com/46951365/104834476-e1423d80-58e2-11eb-9eeb-885e32fb7f20.png' alt='drawing' width='400' style="zoom: 200%;" />

- 핵심 paper
    1. Seq2Seq, or RNN Encoder-Decoder (Cho et al. (2014), Sutskever et al. (2014))
    2. Alignment models (Bahdanau et al. (2015), Luong et al. (2015))
    3. Visual attention (Xu et al. (2015))
    4. Hierarchical attention (Yang et al. (2016))
    5. Transformer (Vaswani et al. (2017))

1. Sequence to sequence (Seq2Seq) architecture for machine translation
    - two recurrent neural networks (RNN), namely encoder and decoder.
    - <img src='https://user-images.githubusercontent.com/46951365/104836546-ebb80380-58f1-11eb-97f7-14818008709f.png' alt='drawing' width='400' style="zoom:200%;" />
    - [RNN,LSTM,GRU](https://excelsior-cjh.tistory.com/185)'s hidden state from the encoder sends source information to the decoder.
    - a fixed-length vector, only last one Hidden stats, Long sentences issue 와 같은 문제점들이 있다. 
    - RNN으로는 Gradient exploding, Gradient vanishing 문제가 크게 일어난다.

2. [Align & Translate](https://arxiv.org/abs/1409.0473)
    - input 데이터에 대해서 information from all hidden states from encoder cells를 저장할 수 있다. 맨뒤, 맨앞의 정보는 사라지는 것이 아닌!
    - at1, at2... atT를 사용해서 X1..XT 중 어떤것에 더 attetion할지의 정보를 담을 수 있다. 
    - 아래의 형태 말고도, 논문에는 더 많은 형태의 모델을 제시해놓았다.
    - <img src='https://user-images.githubusercontent.com/46951365/104836696-0f2f7e00-58f3-11eb-8ad6-b1ac18634d73.png' alt='drawing' width='150' style="zoom:150%;" />

3. [Visual attention](http://proceedings.mlr.press/v37/xuc15.pdf)
    - the image captioning problem 에서 input image와 oput word를 align 하기를 시도 했다.
    - CNN으로 feature를 뽑고, 그 정보를 RNN with attention 사용하였다. 위의 #2글 참조.
    - translation 보다 다른 많은 문제에서 attention을 쉽게 적용한 사례중 하나이다.
    - <img src='https://buomsoo-kim.github.io/data/images/2020-01-01/5.png' alt='drawing' width='300' style="zoom:180%;" />

4. [Hierarchical Attention](https://www.aclweb.org/anthology/N16-1174.pdf)
    - effectively used on various levels
    - attention mechanism 이 classification problem에서도 쓰일 수 있다는 것을 보였다. 
    - 내가 그림만 보고 이해하기로는 아래와 같은 분석을 할 수 있었다.
    - 2개의 Encoder를 사용했다. (word and sentence encoders)
    - <img src='https://user-images.githubusercontent.com/46951365/104837174-2c198080-58f6-11eb-9d4b-b92145a553c0.png' alt='drawing' width='500' style="zoom:150%;" />

5. Transformer and BERT
    1. paper
        - [Attention Is All You Need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
        - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf?source=post_elevate_sequence_page)
    2. multi-head self-attention : 시간 효율적, representation 효율적 따라서 Convolution, recursive operation을 삭제하고, 모두 multi-head attention 모듈로 대체했다. 이 모듈에 대해서는 나중에 공부해볼 예정이다.
    3. BERT 는 pretrains bi-directional representations with the improved Transformer architecture. 그리고 breakthrough results such as XLNet, RoBERTa, GPT-2, and ALBERT 를 가져다 주었다. 

6. [Vision Transformer](https://arxiv.org/pdf/2010.11929.pdf)
    1. Transformer virtually replaces convolutional layers rather than complementing them
    2. CNN’s golden age. Transformer에 의해서 RNN, LSTM 등이 종말을 마지한 것처럼.

7. Conclusion
    - attention will be more and more domains welcoming the application
    - 추천 논문 : [Attention Self-Driving Cars](https://openaccess.thecvf.com/content_ICCV_2017/papers/Kim_Interpretable_Learning_for_ICCV_2017_paper.pdf) 





# 4. How Transformers work intuitively

- The famous paper “Attention is all you need” in 2017 changed the way we were thinking about attention
- Transformer의 기본 block은 self-attention이다. 그러기 위해서는 RNN과 LSTM (were performed sequentially)에 대해서 알아보아야 한다.

1. Representing the input sentence (sequential 한 데이터를 포현(함축)하는 방법)
    - transformer : entire input sequence를 넣어주는거는 어떨까? 단어 단위로 잘라서 (tokenization) sequential 하게 넣어주지 말고!
    - tokenization를 해서 하나의 set을 만든다. 여기에는 단어가 들어가고, 단어간의 order는 중요하지 않다(irrelevant).
    - set 내부 단어들을, project words in a distributed geometrical space. (= word embeddings)
    - **Word Embeddings**
        - character <- word <- sentence와 같이 distributed low-dimensional space로 표현하는 것.
        - 단어들은 각각의 의미만 가지고 있는게 아니라, 서로간의 상관관계를 가지기 때문에 3차원 공간상에 word set을 뿌려놓으면 비슷한 단어는 비슷한 위치에 존재하게 된다. [visualize word Embeddings using t-SNE](https://habr.com/en/company/mailru/blog/449984/)
        - [단어 임배딩이 무엇이고 어떻게 구현하는가 : 특정 단어를 4차원 백터로 표현하는 방법](https://www.tensorflow.org/tutorials/text/word_embeddings)
        - <img src='https://user-images.githubusercontent.com/46951365/104839309-7e14d300-5903-11eb-9f66-9e24a32532b5.png' alt='drawing' width='200' style="zoom:150%;" />
    - **Positional encodings**
        - <img src='https://user-images.githubusercontent.com/46951365/104838732-2de84180-5900-11eb-9c41-813a99d3b8e8.png' alt='drawing' width='250' style="zoom:150%;" />
        - 위에서는 단어의 order를 무시한다고 했다. 하지만 order in a set를 모델에게 알려주는 것은 매우 중요하다.
        - positional encoding : word embedding vector에 추가해주는 set of small constants(작은 상수) 이다. 
            - sinusoidal function (sin함수) 를 positional encoding 함수로 사용했다.
            - sin함수에서 주파수(y=sin(fx), Cycle=2pi/f)와 the position in the sentence를 연관시킬 것이다. 
            - 예를 들어보자. (위의 특정 단어를 4차원 백터로 표현한 그림 참고)
            - 만약 32개의 단어가 있고, 각각 1개의 단어는 512개의 백터로 표현가능하다.
            - 아래와 같이 1개 단어에 대한 512개의 sin(짝수번쨰단어),cos(홀수번째단어)함수 결과값을 512개의 백터 값으로 표현해주면 되는 것이다.
            - <img src='https://user-images.githubusercontent.com/46951365/104839384-0b582780-5904-11eb-878b-79884b0f82f7.png' alt='drawing' width='500' style="zoom:130%;" />
            - 이런 식으로 단어를 나타내는 자체 백터에 에 순서에 대한 정보를 넣어줄 수 있다. 
            - 예를 들어, 영어사전 안에 3000개의 단어가 있다면 3000개의 단어를 각각 2048개의 백터로 표현해 놓는다. 프랑스어사전 안에서 그에 상응하는 3000개의 단어를 추출하고, 단어 각각 2048개의 백터로 임배딩 해놓는다. 그리고 아래의 작업을 한다고 상상해라.
2. Encoder of the Transformer
    - Self-attention & key, value query
        - <img src='https://user-images.githubusercontent.com/46951365/104886249-c0461f00-59ac-11eb-9274-0912fe7d3d76.png' alt='drawing' width='600' style="zoom:200%;" /> 
        - [**매우 추천 동영상, Attention 2: Keys, Values, Queries**](https://www.youtube.com/watch?v=tIvKXrEDMhk) : 위의 [Attention Mechanism](https://junha1125.github.io/artificial-intelligence/2021-01-17-Attention/#2-attention-mechanism-awesome)을 다시 설명해 주면서, key values, Query를 추가적으로 설명해준다.
        - 위의 'What is an attention model' 내용에 weight 개념을 추가해준다. C,yi,si 부분에 각각 Query, key, Value라는 이름의 weight를 추가해준다.
        - [NER = Named Entity Recognition](https://wikidocs.net/30682) : 이름을 보고, 어떤 유형인지 예측하는 test. Ex.아름->사람, 2018년->시간, 파리->나라 
        - 직관적으로 이해해보자. 'Hello I love you'라는 문장이 있다고 치자. Love는 hello보다는 I와 yout에 더 관련성이 깊을 것이다. 이러한 **관련성, 각 단어에 대한 집중 정도**를 표현해 주는 것이, 이게 바로 weight를 추가해주는 개념이 되겠다. 아래의 그림은 각 단어와 다른 단어와의 관련성 정도를 나타내주는 확률 표 이다.
        - <img src='https://user-images.githubusercontent.com/46951365/104888020-8591b600-59af-11eb-816d-567fc49e6cd6.png' alt='drawing' width='250' style="zoom:130%;" />
    - Multi Head Attention
        - [참고 동영상, Attention 3: Multi Head Attention](https://www.youtube.com/watch?v=23XUv0T9L5c)
        - I gave my dog Charlie some food. 라는 문장이 있다고 치자. 여기서 gave는 I, dog Charlie, food 모두와 관련 깊은데, 이게 위에서 처럼 weight 개념을 추가해 준다고 바로 해결 될까?? No 아니다. 따라서 우리는 extra attention 개념을 반복해야 한다. 
        - <img src='https://user-images.githubusercontent.com/46951365/104889635-d5717c80-59b1-11eb-9925-d94e2aee717e.png' alt='drawing' width='600' style="zoom:200%;" />
    - Short residual skip connections
        - 블로그의 필자는, ResNet구조를 직관적으로 이렇게 설명한다. 
        - 인간은 top-down influences (our expectations) 구조를 사용한다. 즉 과거의 기대와 생각이 미래에 본 사물에 대한 판단에 영향을 끼지는 것을 말한다. 
    - Layer Normalization, The linear layer 을 추가적으로 배치해서 Encoder의 성능을 높힌다.
    - <img src='https://user-images.githubusercontent.com/46951365/104891020-bb389e00-59b3-11eb-8b78-3d6e75d64d13.png' alt='drawing' width='250' style="zoom:120%;" />

3. Decoder of the Transformer
    - Input : **Hellow I love you** -> Output : **bonjour je t'aime** (프랑스어로 사랑해)
    - The output probabilities :  **bonjour je t'aime**를 위한 확률 값이 나와야 함. The output probabilities predict the next token in the output sentence.(?)
    - test할때는 output을 넣어주지 않는다. (이 input, output, probabilities는 아직 잘 모르겠다.. 문장이 들어가는 건지, **~~단어~~**(3. What is an attention model?의 사진을 봐라. 단어 하나가 들어가선, 어떤 단어에 더 집중해야 하는가 하는 확률값을 나타낼수가 없다.)가 들어 가는 건지... ㅠㅠ 나중애 필요하면 공부하자)
    - 그냥 Multi-Head Attentoin 을 사용하는게 아니라, Masked Multi-Head Attention 을 사용한다.
    - 블로그는 더 이상 뭔소리 인지 모르겠다. 패스

4. [Attention 4 - Transformers](https://www.youtube.com/watch?v=EXNBy8G43MM)
    - **Multi-head attention is a mechanism to add context that's not based on RNN**
    - <img src='https://user-images.githubusercontent.com/46951365/104905766-23907b00-59c6-11eb-974b-054fa5b4a5f2.png' alt='drawing' width='600' style="zoom:200%;" />
    - <img src='https://user-images.githubusercontent.com/46951365/104905809-31de9700-59c6-11eb-82f5-15e762bf993f.png' alt='drawing' width='600'/>
    - Input에는 문장이 들어가고, Multi-head Attention에서 각 단어에 대한 중요성을 softmax(si)로 계산 후, 그것을 가중치wi로 두고, 원래 단어와 곱해놓는다. (What is an attention model?의 사진 참조)
    - Multi-head Attention **doesn't care**.
    - But Positional Enxoding push to **care more about position**.
    - NLP에서 DIET architecture로 많이 사용중. 
    - Why Multi-head Attention can be replaced RNN
        - <img src='https://user-images.githubusercontent.com/46951365/104907331-47ed5700-59c8-11eb-8a6c-9f57a94b9e19.png' alt='drawing' width='500' style="zoom:150%;" />