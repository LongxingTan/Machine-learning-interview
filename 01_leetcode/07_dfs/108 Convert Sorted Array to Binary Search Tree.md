# 108. Convert Sorted Array to Binary Search Tree
[https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)


## solution

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return None
        mid = len(nums) // 2
        root = TreeNode(val=nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root
```
时间复杂度：O() <br>
空间复杂度：O()


## follow up

[*426. Convert Binary Search Tree to Sorted Doubly Linked List](https://leetcode.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/)
```python
# 中序，记录前一个pre

```
