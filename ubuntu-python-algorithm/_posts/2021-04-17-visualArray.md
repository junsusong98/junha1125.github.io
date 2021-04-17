---
layout: post
title: 【Python】Visualize torch.tensor or numpy.ndarray
description: In this short guide, you’ll see how to Visualize torch.tensor or numpy.ndarray
---

how to Visualize torch.tensor or numpy.ndarray

# Visualize torch.tensor or numpy.ndarray
```python
# from util.npa2csv import Visualarr; Visualarr(x)
import numpy as np
import torch

def Visualarr(arr, out = 'array_out.txt'):
    dim = arr.ndim 
    if isinstance(arr, np.ndarray):
        # (#Images, #Chennels, #Row, #Column)
        if dim == 4:
            arr = arr.transpose(3,2,0,1)
        if dim == 3:
            arr = arr.transpose(2,0,1)
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()
    
    with open(out, 'w') as outfile:    
        outfile.write('# Array shape: {0}\n'.format(arr.shape))
        if dim == 1 or dim == 2:
            np.savetxt(outfile, arr, fmt='%-7.3f')
        elif dim == 3:
            for i, arr2d in enumerate(arr):
                outfile.write('# {0}-th channel\n'.format(i))
                np.savetxt(outfile, arr2d, fmt='%-7.3f')
        elif dim == 4:
            for j, arr3d in enumerate(arr):
                outfile.write('\n# {0}-th Image\n'.format(j))
                for i, arr2d in enumerate(arr3d):
                    outfile.write('# {0}-th channel\n'.format(i))
                    np.savetxt(outfile, arr2d, fmt='%-7.3f')
        else:
            print("Out of dimension!")

    

def test_va():
    arr = np.random.rand(4,2)
    tens = torch.rand(2,5,6,3)
    Visualarr(arr)

test_va()
```





# How to use 

`from pythonfile import Visualarr; Visualarr(x)`

