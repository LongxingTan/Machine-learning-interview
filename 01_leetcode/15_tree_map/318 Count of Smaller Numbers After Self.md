# 318 Count of Smaller Numbers After Self
[https://leetcode.com/problems/maximum-product-of-word-lengths/](https://leetcode.com/problems/maximum-product-of-word-lengths/)


## solution

- 线段树、二分索引树、TreeMap都可以

```python
# bitmask

class Solution:
    def maxProduct(self, words: List[str]) -> int:
        num_words = len(words)
        # Create a list to store the bitmask representation of each word
        masks = [0] * num_words
        
        # Generate a bitmask for each word where bit i is set if the 
        # word contains the i-th letter of the alphabet: [4194311, 33554435, 16416, 131075, 8921120, 63]
        for i, word in enumerate(words):
            for ch in word:
                masks[i] |= 1 << (ord(ch) - ord('a'))

        max_product = 0
        for i in range(num_words - 1):
            for j in range(i + 1, num_words):
                if masks[i] & masks[j] == 0:
                    max_product = max(max_product, len(words[i]) * len(words[j]))

        return max_product
```
时间复杂度：O() <br>
空间复杂度：O()
