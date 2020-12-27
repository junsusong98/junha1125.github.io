---
layout: post
title: 【Pytorch】Pytorch internal에 대한 탐구
# description: >
---

- Pytorch internal에 대한 탐구를 해본다.
- Pytorch와 C와의 연결성에 대해 공부해본다.

# 1. where is torch.tensor.item() defined at the Github?
- stackoverflow에 이와 같이 질문을 남겼다. [stackoverflow LINK](https://stackoverflow.com/questions/65445621/where-is-torch-tensor-item-defined-at-the-github)
- 나는 Link에 나와 있는 것처럼 답변을 받았다.
- 하지만 완벽히 이해를 하지 못해서 공부해서 다시 답변을 이해해 보려고 한다, 
- 그래서 나는 아래의 사이트를 공부하기로 하였다. 
    1. https://pytorch.org/blog/a-tour-of-pytorch-internals-1/
    2. https://pytorch.org/blog/a-tour-of-pytorch-internals-2/
    3. https://medium.com/@andreiliphd/pytorch-internals-how-pytorch-start-211e0d57ad26

# 2. PyTorch internals. How PyTorch start? 
- **torch/csrc/Module.cpp 이 python과 C++를 연결해준다.**
- [Medium Link](https://medium.com/@andreiliphd/pytorch-internals-how-pytorch-start-211e0d57ad26) 
1. import torch를 하는 순간 torch/init.py를 읽고 from torch._C import *를 실행한다. 이 작업을 통해서 Pytorch 속 C 모듈을 실행한다. 
2. torch._C에는 pyi라는 python 파일들이 있고, 이것이 torch/csrc/Module.cpp와 연결되어 있다. 
3. torch/csrc/Module.cpp를 보면 
    - THP란?
        - 간단하게, 리눅스의 메모리 관리 시스템
        - what THP? : https://docs.mongodb.com/manual/tutorial/transparent-huge-pages/
        - TLB란 : https://ossian.tistory.com/38
    0. PyObject 라는 자료형이 원래 어딘가 저장되어 있는 듯 하다. (나중에 VScode에 전체 페키지를 올리고 go to definition 해볼것.)
    1. THPModule_initExtension라는 함수가 정의 되어 있다. 

        ```cpp
        - torch::utils::initializeLayouts(); // torch/csrc/utils/tensor_qschemes.h
        - torch::utils::initializeMemoryFormats();
        - torch::utils::initializeQSchemes();
        - torch::utils::initializeDtypes();
        - torch::tensors::initialize_python_bindings(); // torch/csrc/tensor/python_tensor.h
        ```
        - **이 함수는 파이썬 클래스의 추가적인 초기화(C를 사용한 tensor 맴버함수들을 사용할 수 있게끔 Ex.item())를 위해 사용된다.**
        
    2. Tensor storage(tensor 저장공간)은 이 코드에 의해서 메모리 공간이 생성된다. 

        ```cpp
        auto module = THPObjectPtr(PyImport_ImportModule("torch"));

        THPDoubleStorage_postInit(module);
        THPFloatStorage_postInit(module);
        THPHalfStorage_postInit(module);
        THPLongStorage_postInit(module);
        ...
        ...
        ```
    3. THPModule_initNames 라는 함수에 의해서 메모리 공간 이름이 정의 된다.(?)
    4. **PyObject\* initModule() 이 함수에 의해서, torch._C의 python이 csrc 내부의 C++파일들을 사용할수 있게 만들어준다.** 
    5. static struct PyModuleDef torchmodule 이 부분에 의해서 torch._C 를 위한 메모리 공간을 잡아준다.

# 3. A Tour of PyTorch Internals (Part I)
- [document link](https://pytorch.org/blog/a-tour-of-pytorch-internals-1/)
- 우선 지금 이게 급한게 아니라서 나중에 하도록 하자.

# 4. PyTorch Internals Part II - The Build System
- [document link](https://pytorch.org/blog/a-tour-of-pytorch-internals-2/)
