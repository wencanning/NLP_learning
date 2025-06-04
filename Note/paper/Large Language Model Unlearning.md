# **Large Language Model Unlearning**

## 第一遍阅读

作者介绍了一种新的对齐技术 unlearning,  它的表现比RLHF更好，而且计算所需要的时间更少。同时，作者介绍了三种unlearning可以对LLM对齐生效的情况。作者表示，他们是第一个探索LLM unlearning技术的团队。

## 第二遍阅读

作者使用了 gradient ascent-梯度上生的方法来进行unlearning，目标是让LLM停止生成有害内容

### 梯度上升

在更新参数时，让参数沿着$\mathcal{L}_{fgt}$梯度的方向上增加（正常是沿着梯度的反方向）

- 为什么梯度上升是有效的？
- 什么时候使用交叉熵，什么时候使用KL散度呢？

GA的好处：

- 只需要负样本
- 不需要太多的算力
- 表现比RLHF好
