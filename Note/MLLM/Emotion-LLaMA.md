## Abstraction

part 1： 指出传统的单模态方法不能很好的理解现实世界的复杂情绪表达，而现存的多模态方法在融合语音和细微的面部表情动作方面仍然面临挑战。因此提出了 MERR dataset。

part 2： Emotion-LLaMA。该模型通过特定情绪（emotion-specific encoders）的编码器将音频、视觉和文本输入无缝地整合在一起。通过将特征映射到一个共享空间，采用经过改进的 LLaMA 模型并进行指令调整，得到了Emotion-LLaMA。显著提升了情感识别和推理能力

## 1 Introduction



## 3 **Methodology**

### 3.1 **MERR Dataset Construction**

通过algorithm 1：每个视频是由许多帧组成，我们要通过检查`AUs(Action Units)`找到peak frame，然后产生一系列description。最后集成这些描述来合成上下文，将上下文喂给`LLaMA-3`生成综合性描述。

**peak frame**: 使用`OpenFace toolkit`提取人脸，对每一帧检查并打分`AUs`, 找到累计分数最大的帧

**MERR Dataset**：

- $C_{ved}$ : 通过帧对应的AUs找到对应的文本描述（table 11）
- $C_{vod}$：`MiniGPT-v23`分析峰值帧以提取上下文信息
- $C_{atd}$：`Qwen-Audio4`对音频片段进行处理，提取语音和声调的细微差别，生成与情绪相关的描述
- $C_{ls}$：按我的理解可能是该帧对应的字幕
- $C_{md}$：利用上述描述生成最后综合性的描述 ---> 从多个模态分析情感变化，确保最终生成详细的情感描述注解
  - **Coarse-grained annotation synthesis**: 直接将上述描述代入到特定的模板中
  - **Fine-grained description generation by LLaMA-3:** 将粗粒度注解 加上prompt丢到LLaMA-3生成细粒度注解

![image-20250913154111093](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20250913154111093.png)





##  A MERR Dataset Details

### A.1 Categories

MERR Dataset包含了9种情绪分类：neutral, happy, angry, worried,surprise, sad, fear, doubt, and contempt.但大部分数据集只包含前7种，MERR Dataset之所以突出也是因为把doubt和contempt也包含在内。这两种分类通常不容易收集到足够的样本，并且容易和其它的分类混淆，例如worry和doubt。然而，面部表情和上下文线索的细微差异可以帮助区分两者。

为了准确地对这些复杂的情绪进行分类，**MERR 数据集依赖于丰富的多模态描述**，这些描述能够全面地理解情绪状态及其背景。这些描述超越了简单的分类标签，提供了关于**面部表情**、**肢体语言**、**声音线索**以及影响情绪解读的**环境因素**的详细见解。表 12 展示了 MERR 中使用的多模态描述的模板，展示了针对每个样本所捕捉的不同组成部分。这些描述包括一个视觉表达部分，侧重于与该情绪相关的特定面部动作和动作单元，一个视觉客观部分，描述整个场景和背景，一个音频音调部分，捕捉声音线索和语调，以及一个文本部分，提供转录的言语或对话。





## term

- zero-shot learning：模型在**没有任何示例**的情况下，直接根据指令和其在预训练阶段学到的知识来完成任务。这种能力是 LLM 泛化能力最直接的体现
- one-shot learning：模型在仅提供**一个示例**的情况下，来完成类似的任务。这比零样本学习更进了一步，模型能通过一个例子捕捉到任务的特定格式、风格或模式

- few-shot learning：这是目前最常见的提示工程（prompt engineering）技术之一。通过提供几个例子，可以更明确地引导模型，使其输出更符合期望的格式和内容，从而显著提高任务的准确性。