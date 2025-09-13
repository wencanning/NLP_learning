from unsloth import FastLanguageModel

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
load_in_4bit = True
max_seq_length = 4096
dtype = None

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


from datasets import load_dataset
dataset = load_dataset("reciperesearch/dolphin-sft-v0.1-preference")["train"]
dataset = dataset.map(format_prompt,)

import pprint
row = dataset[1]
print('INSTRUCTION: ' + '=' * 50)
pprint.pprint(row["prompt"])
print('ACCEPTED: ' + '=' * 50)
pprint.pprint(row["chosen"])
print('REJECTED: ' + '=' * 50)
pprint.pprint(row["rejected"])


from unsloth import PatchDPOTrainer
PatchDPOTrainer()


from trl import DPOConfig, DPOTrainer
from unsloth import is_bfloat16_supported


args = DPOConfig(
    max_length=max_seq_length,
    max_prompt_length=max_seq_length // 2,
    max_completion_length=max_seq_length // 2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    beta=0.1,
    logging_steps=1,
    optim="adamw_8bit",
    lr_scheduler_type="linear",
    max_steps=30,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    output_dir="output",
    report_to="none",
)


dpo_trainer = DPOTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=args,
)

dpo_trainer.train()


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
