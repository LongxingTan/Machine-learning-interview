# 721 Accounts Merge
[https://leetcode.com/problems/accounts-merge/](https://leetcode.com/problems/accounts-merge/)


## solution
- [账号合并 - 鼓励师的文章 - 知乎](https://zhuanlan.zhihu.com/p/60635925)


- 并查集
```python

```
时间复杂度：O() <br>
空间复杂度：O()

- 集合
```python
# 注意这里的dict如何解决多个相同的key同时存在字典中: dict[str, list[list[str]]]
# 同时, 一个错误思路是, 每次去历史列表里找，找不到则新加一个list, 可能现在找不到，但未来有别的牵桥搭线就可以找到

class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:  
        from collections import defaultdict
        if not accounts:
            return
        lookup = defaultdict(list)
        res = []
        for account in accounts:
            name = account[0]
            email = set(account[1:])

            lookup[name].append(email)
            for e in lookup[name][:-1]:
                # a\b\c搭桥相连, a:123, b:45, c:15
                # a先进入c, 之后b并入c. 
                if e & email:
                    lookup[name].remove(e)
                    lookup[name][-1].update(e)
        
        for key, val in lookup.items():
            for tmp in val:
                res.append([key] + list(sorted(tmp)))
        return res

    
# class Solution:
#     def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
#         mydict = collections.defaultdict(list)
# 
#         for account in accounts:
#             name = account[0]
#             email = set(account[1:])
# 
#             already_in = False
#             if name in mydict:
#                 for prev_email in mydict[name]:
#                     if prev_email.intersection(set(email)):                        
#                         prev_email.update(email)
#                         already_in = True
#                         break            
#             
#                 if not already_in:                    
#                     mydict[name].append(set(email))
#             else:
#                 mydict[name].append(set(email))                    
#         
#         res = []
#         for key, value in mydict.items():
#             for val in value:
#                 res.append([key] + sorted(list(val)))
#         return res
```

- dfs
```python
class Solution(object):
    def accountsMerge(self, accounts):
        from collections import defaultdict
        visited_accounts = [False] * len(accounts)
        emails_accounts_map = defaultdict(list)
        res = []
        # Build up the graph.
        for i, account in enumerate(accounts):
            for j in range(1, len(account)):
                email = account[j]
                emails_accounts_map[email].append(i)
        # DFS code for traversing accounts.
        def dfs(i, emails):
            if visited_accounts[i]:
                return
            visited_accounts[i] = True
            for j in range(1, len(accounts[i])):
                email = accounts[i][j]
                emails.add(email)
                for neighbor in emails_accounts_map[email]:
                    dfs(neighbor, emails)
        # Perform DFS for accounts and add to results.
        for i, account in enumerate(accounts):
            if visited_accounts[i]:
                continue
            name, emails = account[0], set()
            dfs(i, emails)
            res.append([name] + sorted(emails))
        return res
```

- bfs
```python

```
