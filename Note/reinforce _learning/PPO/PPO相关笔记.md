## Reward Model

1. reward model 是什么？

2. reward model 如何评分？ 

## PPO涉及哪些模型？

-> 表示的是输出层的结构和维度

- reference model(基准模型)-> LM Head(HS*VS)： 作为参考模型，训练模型输出的概率分布不能和基准模型相差过大
- 训练模型 -> LM Head(HS*VS)：结构和基准模型完全一致，PPO训练的目标就是优化训练模型
- reward model -> score head(HS*1) ：对一个**问答序列(将question和answer拼在一起)最后一个token**进行评分。
- state-value model -> value head(HS*1)：根据目前已经生成的序列，预测这个问答序列的期望回报是多少。于reward model不同，它需要对每个token都输出

## 关键函数分析

### get_value

```python
# 计算state value
def get_value(critic, question, answer, shift=True):
    # 将question和answer拼在一起
    input_ids = torch.cat((question, answer), 1)
    # 创建注意力掩码
    attention_mask = input_ids != tokenizer.pad_token_id
    # position_ids 是 ​​Transformer 模型用来表示 token 在序列中的绝对位置​​的索引
    # 通常通过 ​​位置嵌入（Position Embeddings）​​ 来实现
    # cumsum(dim)沿着指定维度作累加求和
    position_ids = attention_mask.cumsum(1) - attention_mask.long()

    input_ids = torch.masked_fill(input_ids, ~attention_mask, 0)

    #[b, lens, 768]
    last_hidden_state = critic.gpt_neox(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids).last_hidden_state

    #[b, lens]
    value = critic.score(last_hidden_state)

    # 提取与回答生成过程相关的状态值
    # state value: 从问题最后一个token位置到序列倒数第二个token
    # V(s_t) 应该预测从状态 s_t（生成第 *t* 个token前的状态）开始的未来累积奖励（即还没有生产第k个token）
    if shift:
        value = value[:, question.shape[1] - 1:-1].squeeze(-1)

    return value
```





### get_logprob

```python
def get_logprob(actor, question, answer):
    input_ids = torch.cat((question, answer), 1)
    attention_mask = input_ids != tokenizer.pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(input_ids, ~attention_mask, 0)

    logits = actor(input_ids=input_ids,
                   attention_mask=attention_mask,
                   position_ids=position_ids).logits

    logits = logits[:, question.shape[1] - 1:-1]
    logits /= 0.7  # temperature

    logprob = logits.log_softmax(dim=-1)
    logprob = logprob.gather(2, answer.unsqueeze(-1)).squeeze(-1)

    return logprob
```

- 数据处理部分和get_value的逻辑类似

- logprob = logprob.gather(2, answer.unsqueeze(-1)).squeeze(-1)。
  - 非常关键的数据处理：从模型输出的对数概率分布中，**只选择实际生成的token（即`answer`中的token）对应的对数概率值**
  - gather(2, answer.unqueeze(-1))在词汇表维度（dim=2）上，根据指定的索引选择对应的值

- 函数最终的输出形状(batch size, seq_length), 表示第i个batch生产第j个token的log_prob
