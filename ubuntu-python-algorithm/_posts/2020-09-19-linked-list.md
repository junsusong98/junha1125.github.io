---
layout: post
title: 【Algorithm】[leetcode] linked-list - swap-nodes-in-pairs
# description: > 
---

문제 : [https://leetcode.com/problems/swap-nodes-in-pairs/](https://leetcode.com/problems/swap-nodes-in-pairs/)

# 이해 이미지
- PS - 리스트를 전체로 바라보지 말아라. 결국 하나하나의 객체일 뿐이다. 
- 항상 1. 자리 바꾸기 2. 연결 바꾸기
- ![note원본 P1](https://user-images.githubusercontent.com/46951365/93664812-93403900-faac-11ea-9fb9-3120c9e4902d.png)

# 나의 코드
```python
# https://leetcode.com/problems/swap-nodes-in-pairs/


class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
head.next.next.next = ListNode(4)

print(head.val, head.next.val, head.next.next.val, head.next.next.next.val, head.next.next.next.next)


def swapPairs(head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if not head or not head.next :
        return head
    
    first, second, third = head, head.next, head.next.next
    head = second
    head.next = first
    print(head.val, head.next.val)
    third = swapPairs(third)
    first.next = third
    return head

head = swapPairs(head)
print(head.val, head.next.val, head.next.next.val, head.next.next.next.val, head.next.next.next.next)
```

# 최적회 상대 코드 
```python
def swapPairs(self, head):

    if not head or not head.next:
        return head
  
    first,second = head, head.next
    
    # 자리 바꾸기 먼저
    third = second.next # 1
    head = second       # 2

    # next 연결 바꿔주기
    second.next = first # 3
    first.next = self.swapPairs(third) # 4
    
    return head
```