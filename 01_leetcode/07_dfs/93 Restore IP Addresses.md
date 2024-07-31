# 93 Restore IP Addresses
[https://leetcode.com/problems/restore-ip-addresses/description/](https://leetcode.com/problems/restore-ip-addresses/description/)


## solution

```python
# 分割类回溯

class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        path = []
        res = []
        self.dfs(s, path, res, start=0)
        return res

    def dfs(self, s, path, res, start):
        if start >= len(s) and len(path) == 4:
            res.append('.'.join(path[:]))
        
        for i in range(start, len(s)):
            temp = s[start: i+1]
            if len(temp) > 1 and temp[0] == '0':
                continue
            elif int(temp) > 255:
                continue    
            elif len(path) > 3:
                break  
            else:
                path.append(temp)
                self.dfs(s, path, res, i+1)
                path.pop()
```
时间复杂度：O(3^4) <br>
空间复杂度：O(1)
