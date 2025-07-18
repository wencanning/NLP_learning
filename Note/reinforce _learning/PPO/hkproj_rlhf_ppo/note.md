# RLHF—PPO

## About Language Model

LLM是一种概率模型，它总是进行next token predication。

### LLM如何generate text？

1. 生成单个词

   ![image-20250714213937693](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20250714213937693.png)

给定prompt，llm输出长为词典大小的一系列概率，代表了每个token被选择的机会有多大

2. 生成完整的文本

LLM每次根据概率选择一个token后会将这次新生成的token加入到prompt中，形成新的prompt。接着重复next token predication。直到生成结束标志token。这个过程被称为自回归(autoregression)

## AL alignment

llm在预训练的过程中学习到了海量的知识，能对我们的每个问题进行问答。但如果我们想让llm作为chat assistant，我们希望llm按照一定的格式进行回答：

- 不能使用冒犯的语言
- 不能进行种族歧视
- 回答尽可能的礼貌

这时候就得进行llm alignment。此时就引出了



