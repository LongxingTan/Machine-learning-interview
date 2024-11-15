# 模型解释性
> 在很多实际应用中，模型可解释性(interpretability)是非常重要的。


## 方法

- 线性方程
  - weight代表了与目标的关系

- 树模型
  - 所有树中被用作分裂特征的次数
  - 每个特征在所有树中被用作分裂特征时，所贡献的增益（即信息增益）
  - 特征在所有树中被用作分裂特征时，所覆盖的样本数

- 特征重要性
  - Boruta
  - LOFO

- SHAP

- LIME

- Eli5


## 参考
- [https://blog.ml.cmu.edu/2020/08/31/6-interpretability/](https://blog.ml.cmu.edu/2020/08/31/6-interpretability/)
- [https://github.com/sicara/tf-explain](https://github.com/sicara/tf-explain)
