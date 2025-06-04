import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import os
import torch
from torch import tensor
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, Trainer

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = 'microsoft/deberta-v3-large'
output_dir = "data/"

data = pd.read_csv(output_dir + "finance_sentiment.csv")
data_sample = data.sample(n=3000, random_state=42)


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.model_max_length = model.config.max_position_embeddings
print("model parameters:" + str(sum(p.numel() for p in model.parameters())))

labels = data_sample["label"].tolist()
labels = [0 if x == 0 else 1 for x in labels] # Labels: 0 -> Negative; 1 -> Positive
# convert labels to one hot vectors
labels = np.eye(2)[labels]
print(labels)


train_data,val_data, train_labels, val_labels = train_test_split(data_sample["text"], labels, test_size=1000/len(data_sample), random_state=42)
dataset = Dataset.from_list([{'text': text, 'labels': label} for text, label in zip(train_data, train_labels)])
val_dataset = Dataset.from_list([{'text': text, 'labels': label} for text, label in zip(val_data, val_labels)])

def tokenize_function(examples):
    return tokenizer(examples['text'])

dataset = dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
print(dataset)

from transformers import TrainerCallback, training_args


class CustomCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        # Assuming the evaluation dataset has 'labels' and 'predictions' fields
        eval_dataloader = kwargs['eval_dataloader']
        model = kwargs['model']

        model.eval()
        correct = 0
        total = 0

        for batch in eval_dataloader:
            inputs = batch['input_ids'].to(args.device)
            labels = batch['labels'].to(args.device)

            with torch.no_grad():
                outputs = model(inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)

            labels = torch.argmax(labels, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        print(f"Evaluation Accuracy: {accuracy:.4f}")


trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=val_dataset,
    args=TrainingArguments(
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
        warmup_steps=10,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim=training_args.OptimizerNames.ADAMW_TORCH,
        learning_rate=5e-5,
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        num_train_epochs=3,
        # report_to="wandb",
        report_to="none",
        group_by_length=True,
        eval_strategy="steps",
        eval_steps=20,
    ),
    callbacks=[CustomCallback()],
)

trainer_stats = trainer.train()