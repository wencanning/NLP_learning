
## 核心思想

paper的核心思想：当student model容量不够，表达能力不足时，如果使用forward KLD，student model容易overestimate teacher model的valid region。

reverse KLD：
$$
\begin{aligned}
\theta = \arg \, \underset \theta \min \, \mathcal{L}(\theta) &= \arg \, \underset \theta \min \, KL[q_{\theta} || p] \\
&= \arg \, \underset\theta\min[- \underset {x~p_x,y~q_{\theta}}{\mathbb{E}}\log\frac{p(y|x)}{q_{\theta}(y|x)}]
\end{aligned}
$$
