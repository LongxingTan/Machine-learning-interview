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
  - Shapley值用于公平分配多个参与者在合作中所带来的收益，机器学习模型解释中，SHAP用于计算每个特征对模型预测结果的贡献

- LIME
  - The overall goal of LIME is to identify an interpretable model over the interpretable representation that is locally faithful to the classifier


- Eli5
  - 工具: https://github.com/eli5-org/eli5


## 参考
- [https://blog.ml.cmu.edu/2020/08/31/6-interpretability/](https://blog.ml.cmu.edu/2020/08/31/6-interpretability/)
- [https://github.com/sicara/tf-explain](https://github.com/sicara/tf-explain)
- [Leveraging AI for efficient incident response](https://engineering.fb.com/2024/06/24/data-infrastructure/leveraging-ai-for-efficient-incident-response/)
