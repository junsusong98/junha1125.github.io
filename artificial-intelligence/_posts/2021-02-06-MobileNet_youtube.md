---
layout: post
title: 【Light_W】Understanding Xception, MobileNet V1,V2 from youtube w/ advice
---

- **논문1** : [MobileNet V1 paper link](https://arxiv.org/abs/1704.04861)  / [Youtube Link1](https://www.youtube.com/watch?v=7UoOFKcyIvM)
- **논문2** : [MobileNet V2 paper link](https://arxiv.org/abs/1801.04381)  / [Youtube Link2](https://www.youtube.com/watch?v=mT5Y-Zumbbw)
- 논문2 : [Xception YoutubeLink](https://www.youtube.com/watch?v=V0dLhyg5_Dw&t=10s)
- **분류** : Object Detection
- **공부 배경** : MobileNetV3 Paper 읽기 전 선행 공부로 youtube 발표 자료 보기
- **선배님 조언**:

  1. Experiment와 Result 그리고 Ablation는 구체적으로 보시나요? 핵심 부분을 읽다가, 핵심이 되는 부분을 캐치하고 이것에 대해서 호기심이 생기거나, 혹은 이게 정말 성능이 좋은가? 에 대한 지표로 확인하기 위해서 본다. 따라서 필요거나 궁금하면 집중해서 보면 되고 그리 중요하지 않은 것은 넘기면 된다. 만약 핵심 Key Idea가 Experiment, Result, Ablation에 충분히 설명되지 않는다면 그건 그리 좋지 않은 아이디어 혹은 논문이라고 판단할 수 있다.
  2. 공부와 삶의 균형이 필요하다. 여유시간이 나서 유투브를 보더라도 감사하고 편하게 보고 쉴줄 알아야 하고, 공부할 때는 자기의 연구분야에 대한 흥미를 가지는 것도 매우 중요하다. 또한 취미나 릴렉스할 것을 가지고 가끔 열심히 공부하고 연구했던 자신에게 보상의 의미로 그것들을 한번씩 하는 시간을 가지는 것도 중요하다. 
  3. 여러 선배들과 소통하고, **완벽하게 만들어가지 않더라도** 나의 아이디어로 무언가를 간단하게 만들어 보고 제안해 보는것도 매우 중요하다. **예를 들어서**, "나에게 이런 아이디어가 있으니, 이 분야 논문을 더 읽어보고 코드도 만들어보고 연구도 해보고 비교도 해보고 실험도 많이 해보고, **다~~ 해보고 나서 당당하게! ㅇㅇ선배님께 조언을 구해봐야겠다!**" 라는 태도 보다는 "나에게 이런 아이디어가 있고... 논문 4개만 더 읽고 **대충 나의 아이디어를 PPT로 만들어서 ㅇㅇ선배님께 보여드려보자! 그럼 뭐 좋은건 좋다고 고칠건 고치라고 해주시겠지...! 완벽하진 않더라도 조금 실망하실지라도 조금 철면피 깔고 부딪혀 보자!!"** 라는 마인드.
  4. 논문. **미리미리 써보고 도전하자. 어차피 안하면 후회한다. 부족한 실력이라고 생각하지 말아라. 어려운 거라고 생각하지 말아라. 일단 대가리 박치기 해봐라. 할 수 있다. 목표를 잡고 도전해보자.**
  5. 강의 필기 PDF는 "OneDrive\21.겨울방학\RCV_lab\논문읽기"
  



<아래 이미지가 안보이면 이미지가 로딩중 입니다>

# 1. Inception Xception GoogleNet
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-01.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-02.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-03.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-04.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-05.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-06.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-07.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-08.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-09.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-10.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-11.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-12.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-13.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-14.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-15.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-16.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-17.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-18.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-19.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-20.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-21.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-22.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-23.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-24.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-25.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-26.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-27.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-28.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-29.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-30.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-31.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-32.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-33.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/inceptionandxception_youtube/inceptionandxception_youtube-34.png?raw=true)



# 2. MobileNetV1
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-01.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-02.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-03.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-04.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-05.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-06.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-07.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-08.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-09.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-10.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-11.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-12.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-13.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-14.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-15.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-16.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-17.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-18.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-19.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-20.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-21.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-22.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-23.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenet_youtube/mobilenet_youtube-24.png?raw=true)




# 3. MobileNetV2
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-01.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-02.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-03.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-04.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-05.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-06.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-07.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-08.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-09.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-10.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-11.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-12.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-13.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-14.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-15.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-16.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-17.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-18.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-19.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-20.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-21.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-22.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-23.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-24.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-25.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-26.png?raw=true)
![img](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2021-1/mobilenetv2_youtube/mobilenetv2_youtube-27.png?raw=true)







