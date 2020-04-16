---
layout: post
title: (강화학습,논문) DQN - playing Atari, Human-level control 논문 리뷰
# description: > 
    
---
 

연구실에서 매주 하는 논문 리뷰의 발표를 위해, 

(2013)Playing Atari with Deep Reinforcement Learning

https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

(2015.1)Human-level control through deep reinforcement

https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf

2개의 논문을 읽고 발표를 준비했습니다. 

아래 슬라이드에서 홍색 배경의 슬라이드는 playing atari 논문에서 읽은 내용이고, 

청색 배경의 슬라이드는 human-level control 논문에서 읽은 내용입니다.   

***

![img](https://k.kakaocdn.net/dn/5e9Bf/btqB5helW5K/B8gbZKEsI9BiL4VKXNj1n0/img.jpg)

![img](https://k.kakaocdn.net/dn/dJT9yA/btqB4ZkE9Lq/SCudzPkYjr8Hw4XtfRsBwK/img.jpg)

![img](https://k.kakaocdn.net/dn/lYkSp/btqB6iw4oJV/dnGuewF8k23kqMNfSTKuKk/img.jpg)

![img](https://k.kakaocdn.net/dn/VTbrt/btqB7bRwli9/V50Ci6PZUf2FFk3YWRv8v1/img.jpg)

![img](https://k.kakaocdn.net/dn/cT6Bbr/btqB40qmmUF/EKFmxCGPGKLRQEuKAXvpG1/img.jpg)

![img](https://k.kakaocdn.net/dn/bJuy89/btqB5iqKPx5/RNqKzUwoQlizl8YS1Pj3k1/img.jpg)

![img](https://k.kakaocdn.net/dn/ebpfWc/btqB2IcVHkk/5eTAa3JSbteC9q2CkGO2IK/img.jpg)

![img](https://k.kakaocdn.net/dn/bk7FQ3/btqB4ZyffuR/xuEOzShoed9rSR7O02Z1fK/img.jpg)

![img](https://k.kakaocdn.net/dn/qYWpA/btqB6IWzGEj/QK0i9XiUCigs3bvU9TzBxk/img.jpg)

![img](https://k.kakaocdn.net/dn/bhMsbF/btqB5F0kP1j/gMHNXoh93pSa42hUta2hX0/img.jpg)

![img](https://k.kakaocdn.net/dn/LzKhV/btqB3p5dyl9/J4q4rK8FeSkRYPJ6lIX85K/img.jpg)

![img](https://k.kakaocdn.net/dn/bGUYqG/btqB3XAHgvn/iGcNYnzkwrkwByesUZ3so1/img.jpg)

![img](https://k.kakaocdn.net/dn/Plr2n/btqB5hL870n/FH4JxzoGXnEQv8UKx1YEA1/img.jpg)

![img](https://k.kakaocdn.net/dn/pkRkP/btqB6JgTLAE/N9tH8WY8QRDtYgEqxlnGq1/img.jpg)

![img](https://k.kakaocdn.net/dn/sDiJB/btqB3qwkAmV/dl3tBtLkuEVb5oAaT7OB5k/img.jpg)

![img](https://k.kakaocdn.net/dn/VZOkr/btqB5GLFAFz/sNLBUgUEMhSxhpwa88ooC1/img.jpg)

![img](https://k.kakaocdn.net/dn/bzmjO5/btqB2IcVHqi/L72LIgeashHqXUgevKK0fK/img.jpg)

![img](https://k.kakaocdn.net/dn/qbcRL/btqB5E1qMqh/CHg79C3WmDOKKS6aoBYlQK/img.jpg)

![img](https://k.kakaocdn.net/dn/mpE5B/btqB2I432ij/IbhmsnxFvkSVmMECISI1M1/img.jpg)

![img](https://k.kakaocdn.net/dn/brq7FM/btqB6iX7xhS/PdykaN8cOaQhKqGIVdW5K1/img.jpg)

![img](https://k.kakaocdn.net/dn/cLIJso/btqB7aSDVRJ/Z6iKzYKaRnFmsEkhNr3Ar0/img.jpg)

![img](https://k.kakaocdn.net/dn/ca8XAs/btqB7cbO3Uk/dT74VfYTSzqKI4qvzaKJE1/img.jpg)

![img](https://k.kakaocdn.net/dn/HGcSX/btqB6hZeO3R/jSWuqj0FxxUjkkj6NkgoYK/img.jpg)

![img](https://k.kakaocdn.net/dn/NwSei/btqB6YZbrmQ/zvDJ9AhYpKVUfHnalqQ6O0/img.jpg)

![img](https://k.kakaocdn.net/dn/F7SYV/btqB7bxelpU/VLBzaiCNLnPvcmj0SkAQ01/img.jpg)