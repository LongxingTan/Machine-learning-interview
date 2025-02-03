# 多任务学习
> ML design的排序部分经常需要考虑到多任务学习，有多个loss的其实就算多任务学习
> - 优化策略: loss量级Magnitude, loss学习速率, loss梯度冲突


## 模型

**shared bottom**
- pro: 对于稀疏标签，互相补充学习，任务相关性越高，模型的loss可以降低到更低
- con: 任务之前没有好的相关性时，这种Hard parameter sharing会损害效果

**MOE**
- Soft parameter sharing
- con: MMOE中所有的Expert是被所有任务所共享的，这可能无法捕捉到任务之间更复杂的关系，从而给部分任务带来一定的噪声; 不同的Expert之间没有交互，联合优化的效果有所折扣


**MMOE**
- k个任务则适用k个门控网络

**ESMM**


**PLE**


## 平衡多个任务

**GradNorm**


**DWA (Dynamic Weight Average)**


## Reference
- [收藏|浅谈多任务学习（Multi-task Learning） - 多多笔记的文章 - 知乎](https://zhuanlan.zhihu.com/p/348873723)
- [多任务学习优化（Optimization in Multi-task learning） - 凉爽的安迪的文章 - 知乎](https://zhuanlan.zhihu.com/p/269492239)
- [深度学习的多个loss如何平衡？ - 王晋东不在家的回答 - 知乎](https://www.zhihu.com/question/375794498/answer/2307552166)
- [多目标学习在推荐系统的应用(MMOE/ESMM/PLE) - 绝密伏击的文章 - 知乎](https://zhuanlan.zhihu.com/p/291406172)
- [「算法面试开挂专题」MTL多目标训练20问 - 琦琦的文章 - 知乎](https://zhuanlan.zhihu.com/p/701647428)
- [多场景多任务学习在美团到店餐饮推荐的实践](https://tech.meituan.com/2023/03/23/recommendation-multi-scenario-task.html)
