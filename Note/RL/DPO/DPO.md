# DPO(Direct Preference Optimization)

概述：DPO通过求出object function的解析解，带入到reward model loss中，从而达到**直接偏好优化**的目的，即：跳过训练reward model的过程

## Reward model and Bradley-Terry model

### 偏好数据集

我们训练reward model并不是直接给定一个prompt-answer和对应的reward让model去学习。因为打出一个公正的reward其实是一个比较困难的事情：因为每个人的观点不一样，你觉得好的地方其他人可能不这么觉得。但是，人们擅长比较：我们可以给定两个不同回答，让model学习两个回答的优劣。

因此在train reward时，我们使用的是偏好数据集(preference dataset)。

![image-20250717160227482](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20250717160227482.png)

### Bradley Terry model

Bradley Terry model的核心公式：即winner大于loser的概率
$$
P(y_w > y_l)=\frac{e^{r^*(x,y_w)}}{e^{r^*(x,y_w)}+e^{r^*(x,y_l)}}
$$
我们的目标是最大化这个概率，让winner打败loser的概率尽可能的大：让reward model能够正确的rank the response。接下来推导reward model的loss

首先原式可以用sigmoid来表达：
$$
\frac{e^A}{e^A + e^B} = \frac{1}{1+e^{-(A-B)}}= \sigma(A-B)
$$
 那么我们的Loss可以设计为：
$$
L=-\mathbb{E}_{(x,y_w,y_l) \sim D}[\log \sigma (r(x,y_w) -r(x,y_l))]
$$
前面的负号是因为我们要最大化这个期望。当期望最大化时只可能是winner的reward大于loser的reward

### RLHF objective

在RLHF中，我们的objective总是不变的：最大化累计奖励，并且不偏离最初的模型太多
$$
J_{RLHF} = \max_{\pi_\theta}\mathbb{E}_{x \sim D,y \sim \pi_{\theta(y|x)}}\left[ r_{\phi}(x,y)-\beta\mathbb{D}_{KL}\left[\pi_{\theta}(y|x)||\pi_{\text{ref}}(y|x)\right]\right]
$$
通过计算，我们可以得到该object的解析解：
$$
\pi_r(y|x)=\frac{1}{Z(X)}\pi_{\text{ref}}(y|x)\text{exp}(\frac{1}{\beta}r(x,y))
$$
 然后我们将$r(x,y)$提出单独移到一边:
$$
r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\log Z(x)
$$
最后我们将上述reward公式带入到reward loss中，发现可以刚好消除Z(x)。此时我们观察loss function，该式已经不再包含reward！

![image-20250717170109608](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20250717170109608.png)
