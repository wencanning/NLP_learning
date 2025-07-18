import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported

from trl import CPOConfig, CPOTrainer
from datasets import load_dataset
from typing import List, Literal, Optional

# Model Loading and Configuration

# Set basic parameters
max_seq_length = 4096  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# Load the pre-trained model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-0.5B-Instruct",  # Choose ANY! eg mistralai/Mistral-7B-Instruct-v0.2
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token="hf_...",  # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# Add proper chat template if missing
if tokenizer.chat_template is None:
    DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE


# Dataset Preparation and Processing

# Function to apply chat template
def apply_chat_template(
    example, tokenizer, task: Literal["sft", "generation", "rm", "kto"] = "sft", assistant_prefix="<|assistant|>\n"
):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True if task == "generation" else False
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if chosen_messages[0]["role"] != "system":
                chosen_messages.insert(0, {"role": "system", "content": ""})
            if rejected_messages[0]["role"] != "system":
                rejected_messages.insert(0, {"role": "system", "content": ""})
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
            prompt_messages = [[msg for msg in example["chosen"] if msg["role"] == "user"][0]]
            # Insert system message
            if example["chosen"][0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": ""})
            else:
                prompt_messages.insert(0, example["chosen"][0])
            # TODO: handle case where chosen/rejected also have system messages
            chosen_messages = example["chosen"][1:]
            rejected_messages = example["rejected"][1:]
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            example["text_chosen"] = _strip_prefix(example["text_chosen"], assistant_prefix)
            example["text_rejected"] = _strip_prefix(example["text_rejected"], assistant_prefix)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "kto":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            prompt_messages = [[msg for msg in example["chosen"] if msg["role"] == "user"][0]]
            chosen_messages = prompt_messages + [msg for msg in example["chosen"] if msg["role"] == "assistant"]
            rejected_messages = prompt_messages + [msg for msg in example["rejected"] if msg["role"] == "assistant"]
            if "system" in example:
                chosen_messages.insert(0, {"role": "system", "content": example["system"]})
                rejected_messages.insert(0, {"role": "system", "content": example["system"]})
            example["text_chosen"] = _strip_prefix(tokenizer.apply_chat_template(chosen_messages, tokenize=False), assistant_prefix)
            example["text_rejected"] = _strip_prefix(tokenizer.apply_chat_template(rejected_messages, tokenize=False), assistant_prefix)
        else:
            raise ValueError(f"Could not format example as dialogue for `kto` task!")
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo', 'kto']}"
        )
    return example

# Load the dataset
raw_datasets = load_dataset("trl-lib/ultrafeedback_binarized")
train_dataset = raw_datasets["train"]

# Take a subset of the training data, I only use 1000 examples here
train_subset = train_dataset.select(range(1000))  # Use first 1000 examples for testing purpose. Comment out to use the complete dataset

# Model Training Setup

# Configure the LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)


# Set up the SimPO trainer with appropriate training arguments
simpo_trainer = CPOTrainer(
    model=model,
    args=CPOConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=5e-7,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        output_dir="outputs",
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        seed=42,
        report_to="none", # Use this for WandB etc
        loss_type="simpo",
        cpo_alpha=0.0
    ),
    train_dataset=train_subset, #Use 1000 examples for test
    #train_dataset=train_dataset, #Use the complete dataset
    processing_class=tokenizer,
)


# Model Training Setup

# Configure the LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

training_args = CPOConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-7,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    output_dir="outputs",
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    seed=42,
    report_to="none", # Use this for WandB etc
    loss_type="simpo",
    cpo_alpha=0.0
)

# Set up the SimPO trainer with appropriate training arguments
simpo_trainer = CPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_subset, #Use 1000 examples for test
    #train_dataset=train_dataset, #Use the complete dataset
    processing_class=tokenizer,
)

simpo_trainer.train()