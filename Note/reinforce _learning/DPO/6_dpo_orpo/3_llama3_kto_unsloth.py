from unsloth import FastLanguageModel

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


model_name = "/home/wangjin/models/Meta-Llama-3.1-8B-bnb-4bit"
load_in_4bit = True
max_seq_length = 4096
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    # [NEW]
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)


# Add proper chat template if missing
if tokenizer.chat_template is None:
    DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

# The data must be formatted with appropriate prompt template first.
# See details here: https://github.com/huggingface/trl/blob/main/examples/scripts/orpo.py
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

def format_prompt(sample):
    instruction = sample["instruction"]
    input = sample["input"]
    accepted = sample["accepted"]
    rejected = sample["rejected"]

    # ORPOTrainer expects prompt/chosen/rejected keys
    sample["prompt"] = alpaca_prompt.format(instruction, input, "")
    sample["chosen"] = accepted + EOS_TOKEN
    sample["rejected"] = rejected + EOS_TOKEN
    return sample

from typing import List, Literal, Optional
from datasets import load_dataset

import re
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

# Load the KTO dataset
raw_datasets = load_dataset("trl-lib/kto-mix-14k")
train_dataset = raw_datasets["train"]

# Take a subset of the training data, I only use 1000 examples here
train_subset = train_dataset.select(range(1000))  # Use first 1000 examples

from trl import KTOConfig, KTOTrainer
from unsloth import is_bfloat16_supported


args = KTOConfig(
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
    report_to="none",  # Use this for WandB etc
)


kto_trainer = KTOTrainer(
    model=model,
    train_dataset=train_subset,
    tokenizer=tokenizer,
    args=args,
)

kto_trainer.train()


# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.", # instruction
        "1, 1, 2, 3, 5, 8", # input
        "", # output - leave this blank for generation!
    )
], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
print(tokenizer.batch_decode(outputs))
