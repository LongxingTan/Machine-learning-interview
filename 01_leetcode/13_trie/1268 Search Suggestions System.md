# 1268 Search Suggestions System
[https://leetcode.com/problems/search-suggestions-system/](https://leetcode.com/problems/search-suggestions-system/)


## solution

```python
class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        res = []
        products.sort()

        for i in range(len(searchWord)):
            search = searchWord[:i+1]
            cur = []
            count = 0
            for prod in products:
                if search == prod[:i+1]:
                    cur.append(prod)
                    count += 1

                if count == 3:
                    break
            res.append(cur)
        return res
```
时间复杂度：O() <br>
空间复杂度：O()


- binary search
```python

```
