import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from unsloth import tokenizer_utils, FastLanguageModel
import torch
import datasets
import pandas as pd
from trl import SFTTrainer

from typing import Any, Dict, List, Union
from transformers import DataCollatorForLanguageModeling, TrainingArguments

from sklearn.model_selection import train_test_split

# needed as this function doesn't like it when the lm_head has its size changed
# def do_nothing(*args, **kwargs):
#     pass
# tokenizer_utils.fix_untrained_tokens = do_nothing

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+


# model_name = "unsloth/Qwen2-7B-bnb-4bit";
# model_name = "unsloth/llama-3-8b-bnb-4bit"
model_name = "/home/wangjin/models/Meta-Llama-3.1-8B-bnb-4bit"
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    load_in_4bit = load_in_4bit,
    max_seq_length = max_seq_length,
    dtype = dtype,
)

yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
# keep only the yes and no tokens from lm_head
par = torch.nn.Parameter(torch.vstack([model.lm_head.weight[no_token_id, :], model.lm_head.weight[yes_token_id, :]]))
print(par.shape)
print(model.lm_head.weight.shape)
model.lm_head.weight = par


from peft import LoftQConfig

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        # "lm_head", # can easily be trained because it has only 2 tokens
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = True,  # We support rank stabilized LoRA
    # init_lora_weights = 'loftq',
    # loftq_config = LoftQConfig(loftq_bits = 4, loftq_iter = 1), # And LoftQ
)

print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# load dataset
data = pd.read_csv(os.path.join('data', "finance_sentiment.csv")) # columns are text,label

train_size = 1000
val_size = 1000

# keep a subset (for testing)
data_sample = data.sample(n=train_size+val_size, random_state=42)


train_df, val_df = train_test_split(data_sample, test_size=val_size/len(data_sample), random_state=42)
print(len(train_df))


train_dataset = datasets.Dataset.from_pandas(train_df, preserve_index=False)

prompt = """Here is a financial tweet:
{}

Does this tweet have a positive sentiment? Answer with "Yes" or "No".

SOLUTION
The correct answer is: "{}"""

positivelabel = "Yes"
negativelabel = "No"


def formatting_prompts_func(dataset_):
    # this is to fix an issue with a certain transformers version, you might not need this
    if isinstance(dataset_['text'], str):
        if model_name.lower().__contains__("qwen"):
            return [""] * 100
        elif model_name.lower().__contains__("llama"):
            return " "
        else:
            return " "

    texts = []
    for i in range(len(dataset_['text'])):
        t = dataset_['text'][i]
        label = positivelabel if dataset_['label'][i] == 1 else negativelabel
        text = prompt.format(t, label)

        texts.append(text)
    return texts


# this custom collator is needed to change the sequence labels from yes_token_id and no_token_id to 1 and 0. It also trains only on the last token of the sequence.
class DataCollatorForLastTokenLM(DataCollatorForLanguageModeling):
    def __init__(
        self,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)
        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        for i in range(len(examples)):
            # Find the last non-padding token
            last_token_idx = (batch["labels"][i] != self.ignore_index).nonzero()[-1].item()
            # Set all labels to ignore_index except for the last token
            batch["labels"][i, :last_token_idx] = self.ignore_index
            # The old labels for the Yes and No tokens need to be mapped to 1 and 0
            batch["labels"][i, last_token_idx] = 1 if batch["labels"][i, last_token_idx] == yes_token_id else 0


        return batch


collator = DataCollatorForLastTokenLM(tokenizer=tokenizer)


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # not needed because group_by_length is True
    args = TrainingArguments(
        per_device_train_batch_size = 32,
        gradient_accumulation_steps = 1,
        warmup_steps = 10,
        learning_rate = 1e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        num_train_epochs = 1,
        # report_to = "wandb",
        report_to = "none",
        group_by_length = True,
    ),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # not needed because group_by_length is True
    args = TrainingArguments(
        per_device_train_batch_size = 32,
        gradient_accumulation_steps = 1,
        warmup_steps = 10,
        learning_rate = 1e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        num_train_epochs = 1,
        # report_to = "wandb",
        report_to = "none",
        group_by_length = True,
    ),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)


#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()


#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


FastLanguageModel.for_inference(model) # Enable native 2x faster inference

from collections import defaultdict
import torch.nn.functional as F

# Step 1: Tokenize the inputs and sort them by their tokenized length
tokenized_inputs = []
for i in range(len(val_df['text'])):
    text = val_df['text'].iloc[i]
    test_str = prompt.format(text, "")
    tokenized_input = tokenizer(test_str, return_tensors="pt", add_special_tokens=False)
    tokenized_inputs.append((tokenized_input, test_str, val_df['label'].iloc[i]))

# Sort by tokenized length
tokenized_inputs.sort(key=lambda x: x[0]['input_ids'].shape[1])

# Step 2: Group the inputs by their tokenized length
grouped_inputs = defaultdict(list)
for tokenized_input, test_str, label in tokenized_inputs:
    length = tokenized_input['input_ids'].shape[1]
    grouped_inputs[length].append((tokenized_input, test_str, label))

# Step 3: Process each group in batches of 64
batch_size = 64
all_outputs = []
all_strings = []
all_labels = []

from tqdm import tqdm

for length, group in tqdm(grouped_inputs.items()):
    for i in range(0, len(group), batch_size):
        batch = group[i:i + batch_size]
        batch_inputs = [item[0] for item in batch]
        batch_strings = [item[1] for item in batch]
        batch_labels = [item[2] for item in batch]

        # Concatenate the batch inputs
        input_ids = torch.cat([item['input_ids'] for item in batch_inputs], dim=0).to("cuda")
        attention_mask = torch.cat([item['attention_mask'] for item in batch_inputs], dim=0).to("cuda")

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # print(outputs.logits[:, -1].shape)

        # Get logits for the first token prediction (assuming binary classification)
        logits = outputs.logits[:, -1, :2]  # Only consider logits for 0 and 1

        # Apply softmax
        probabilities = F.softmax(logits, dim=-1)

        # Get predictions
        predictions = torch.argmax(probabilities, dim=-1)

        all_outputs.extend(predictions.cpu().numpy())
        all_labels.extend(batch_labels)
        all_strings.extend(batch_strings)

# Step 4: Do the label assignment
correct = 0
total = 0

for i in range(len(all_outputs)):
    pred = str(all_outputs[i])
    label = str(all_labels[i])
    if i > len(all_outputs) - 25:
        print(f"{i}: text: {all_strings[i]}\n pred: {pred} label: {label}\n")

    if pred == label:
        correct += 1
    total += 1

print(f"Correct: {correct} Total: {total} Accuracy: {correct / total}")