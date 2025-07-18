## 学习LLM过程中的疑惑

### 1. 在STF的过程中chat template的作用是什么？什么时候需要使用chat template

模型分为base model和 instruct model

A base model is trained on raw text data to predict the next token, while an instruct model is fine-tuned specifically to follow instructions and engage in conversations. For example, [`SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) is a base model, while [`SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) is its instruction-tuned variant.

```
When using an instruct model, always verify you're using the correct chat template format. Using the wrong template can result in poor model performance or unexpected behavior. The easiest way to ensure this is to check the model tokenizer configuration on the Hub. For example, the `SmolLM2-135M-Instruct` model uses this configuration.
```



- 作用：将原始对话数据（如多轮对话）转换为模型可处理的标准化格式

```
# 原始对话
["用户: 你好", "助手: 你好！需要帮助吗？"]

# 应用模板后（以 Llama-2 为例）
"[INST] 你好 [/INST] 你好！需要帮助吗？"
```

- 何时使用： 当训练数据是对话形式（如客服日志、多轮问答），必须通过模板将数据转为模型支持的格式