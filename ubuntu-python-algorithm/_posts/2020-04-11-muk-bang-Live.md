---
layout: post
title: (Algo) 무지의 먹방 라이브 - 카카오문제
description: > 
    프로그래머스 무지의 먹방 라이브 문제 풀이
---

문제링크 : [무지의 먹방라이브](https://programmers.co.kr/learn/courses/30/lessons/42891)
## **문제 설명과 최소값 지우기 방법으로 직접 해보기**
![image](https://user-images.githubusercontent.com/46951365/79040791-0c964e80-7c26-11ea-94c5-760a1aa3249d.png)
![image](https://user-images.githubusercontent.com/46951365/79117750-ed82f280-7dc6-11ea-823f-0e44d710e407.png)

# 나의 고민 과정  
- ## Key point
1. k초 직후 = **이미** k번 음식을 씹어 먹었다.
2. 리스트 내부 최솟값 이용할 떄 처리 순서 : 
    + 리스트 내부 원소 지우기 
    + 나머지 모든 원소에 값 빼기  

- ## 질문
1. index가 포함된 새로운 리스트를 만들 필요가 있을까?
2. 튜플, 딕셔너리, set 무엇을 사용하는게 좋을까?
3. list는 건들지 않고 k을 더하거나 뺴는 방법은 없을까?
    - 예를 들어, [3 2 2] ---6초 직후--> [1 0 0]  
      == [3 2 2]는 그대로 놔두고, k - 6 ...??? 는 아닌것 같다. 
4. 최솟값 말고 다른 방법은 없을까? 
    - 직접 해보면서 문제점을 찾아보자. -->  우선 나쁘지 않은 듯 하다.  
        -> ***이렇게 내가 컴퓨터다. 라는 마음으로 직접하면서 많이 배운다!!***  
    - 큐? 스택? 다이나믹? 정렬? 
5. 정렬을 이용하는 방법은 딕션? 튜플? 무엇을 이용하는게 좋을까? 
    - 3번째 방법의 2번째 함수는 무조건 2000보다 작은 시간복잡도를 가지므로, 굳이 딕션이나 튜플을 이용하지 않아도 될 것같다.
    - 모든 수의 합을 구해, -1인지 아닌지를 가장 먼저 판별하는 행위는 2000의 시간복잡도를 가진다. 이것을 가장 앞에 추가하는게 과연 좋을까? 
    - sort말고 len(food_times)의 시간복잡도를 가지는 방법이 없을까??
    - 굳이 sort하는 list를 또 만들어야 할까?? (공간 복잡도 최소화)


- ## 해결방법(2번과 3번 이용)
1. 정직하게 while문을 돌면서, 리스트의 원소들에 -1을 해가는 방법
2. 리스크의 최솟값(가장 빨리 다 먹는 음식)을 찾는다=t   
    - t  *  len(food_times)(=남은 음식 갯수) =< k 이면,  
        - 최솟값을 찾는 행동 계속한다. reculsive하게... 
    - else :
        - 최솟값을 찾는 행동 멈춘다. 
    - 이 방법에는 Dictionary를 이용하는게 낫겠다.  
    - 정확성 테스트 통과 but **효율성에서 시간 초과**. 따라서 이 방법은 적절하지 않다.
3.  정렬을 적극 이용하기 - 정렬만 최대 2000 * log(2000) = 2000 * 10 의 시간 복잡도.
    - 우선 food_times를 정렬한다.
    - 작은 수부터 k를 처리해 나간다.(처리하는 방법은 2번문제 해결법을 적극 이용한다)  
    - k값의 연산은 하지만, 리스트 자체의 연산은 하지 않기 때문에 경과시간이 적을 듯 하다.


# 손코딩

## 2번째 해결방법을 이용한 풀이 (시간초과)
- 2개의 함수를 만든다  
    - Stop을 위한 함수 
    - reculsive를 위한 함수 (최솟값을 찾는다)  

0. sum(food_times) =< k 이면, return -1 
1. food_times값을 value로 하여, 딕셔너리에 값을 채워 넣는다.
2. while -> dick의 value 중 min값 &nbsp;&nbsp; **VS** &nbsp;&nbsp; k 값  -> 크기 비교
    - if. min*원소갯수 =< k :
        - min인 원소 제거
        - 나머지 원소들의 value - min
        - k = k - min*원소갯수
    - else. min*원소갯수 > k : 
        - (k % 원소갯수)의 그 다음 key 를 return  


## 3번째 해결방법을 이용한 풀이 (통과!)
- 2개의 함수를 만든다  
    - cur_min, cur_pos 를 뽑아주는 함수
        - input : list, cur_pos, cur_min, n
        - cur_pos부터 cur_min보다 큰 수를 찾는다.
        - 큰 수를 찾았다면,return : cur_min 과 cur_pos
        - for 문을 돌아 n이 될 때까지, 보다 큰 수를 못찾았다면, cur_pos에 2001 return
    - cur_min보다 같거나 큰 수 중, (k)%n 번째 음식의 index를 찾아주는 함수.
        - input : food_times, (k)%n, cur_min
        - for 문을 돌면서, cur_min보다 같거나 큰 수가 나오면 (k)%n - 1을 한다.
        - -1을 하기 전에 (k)%n가 0이면, 그 때의 food_times의 index를 return한다. 

0. (-1을 return하는 경우가 필요한지는 실험을 통해 알아보자.)
1. food_times를 sort한다.
2. cur_min = list[0]; cur_pos  = 0; n = len(food_times);
3. while cur_min * n <= k :  
    - **diff라는 개념! food_times간의 차이** <- 실수해서 초반 오답 해결: 손코딩한거 직접 예제 돌려보자. 
    - k, cur_min과 cur_pos를 갱신
    - cur_pos가 2001이면 -1 return하고 함수 종료
4. 위의 while이 break되면,
    cur_min의 값보다 같거나 큰 수들 중, (k)%n 번쨰 음식을 찾는다.

# 코드
## 2번째 풀이 코드
```python
def renew(m,dic,k) :
    # k값 갱신
    k = k - m*len(dic)
    keylist = []
    # dic 갱신
    for key, value in dic.items():
        if value == m :
            keylist.append(key)
        else :
            dic[key] = value - m
    for i in keylist:
        del(dic[i])

    return k

def solution(food_times, k):
    # 0
    if sum(food_times) <= k : return -1

    # 1
    dic  = {}
    for index, time in enumerate(food_times):
        dic[index+1] = time
    
    # 2
    while(1):
        m = min(dic.values())
        if m*len(dic) > k : break
        else : k = renew(m,dic,k)

    # 3
    answer = list(dic.keys())[k%len(dic)]
    return answer

if __name__ == "__main__":
    food_times = [3,1,2]
    k = 5
    print(solution(food_times,k))
```

## 3번째 풀이 코드 
```python
def cur_next(lis, cur_min, cur_pos, n, l):
    for i in range(cur_pos+1, l):
        if cur_min < lis[i]:
            return lis[i], i
    return -1 , l+1

def fine_index(food_times, m, cur_min) : 
    for index, Value in enumerate(food_times):
        if cur_min <= Value :
            if m == 0:
                return index+1
            else :
                m -= 1

def solution(food_times, k):
    # 1
    lis = food_times[:]
    lis.sort()
    # 2
    cur_min = lis[0]
    diff = cur_min
    cur_pos = 0 # list 0부터 시작
    n = len(lis) # 남은 음식 수
    l = n # 처음 음식의 총 갯수
    # 3
    while diff*n <= k:
        k = k - diff*n
        temp, cur_pos = cur_next(lis, cur_min, cur_pos, n, l)
        if temp == -1 : return -1 # k가 충분히 커, 음식을 다 먹었을 경우.
        diff = temp - cur_min
        cur_min = temp
        n = l - cur_pos
        
    # 4
    cur_min = lis[cur_pos]
    answer = fine_index(food_times, k%n, cur_min)
    return answer

if __name__ == "__main__":
    food_times = [3,1,2,2,3,2]
    k = 13
    print(solution(food_times,k))
```