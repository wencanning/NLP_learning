import torch
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = 'left'

print(tokenizer)

from datasets import load_dataset, concatenate_datasets

dataset = load_dataset('imdb')
dataset = concatenate_datasets(list((dataset.values())))

f = lambda data: {
    'input_ids': tokenizer.encode(data['text'], truncation=True, max_length=5)
}
dataset = dataset.map(f, remove_columns=dataset.column_names)

dataset = dataset.train_test_split(test_size=2000)

print(dataset, dataset['train'][0])


from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

model_actor = AutoModelForCausalLM.from_pretrained('model/actor')
model_actor_ref = AutoModelForCausalLM.from_pretrained('model/actor')

model_critic = AutoModelForSequenceClassification.from_pretrained(
    'model/critic', num_labels=1)
model_critic_ref = AutoModelForSequenceClassification.from_pretrained(
    'model/critic', num_labels=1)


from trl.trainer.ppov2_trainer import PPOv2Config, PPOv2Trainer
from transformers import TrainerCallback

config = PPOv2Config(
    output_dir='output_dir',
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    total_episodes=20_0000,
    learning_rate=5e-6,
    logging_dir='output_dir',
    run_name='run_name',
    #non_eos_penalty=True,
    save_strategy='no')

trainer = PPOv2Trainer(config=config,
                       tokenizer=tokenizer,
                       policy=model_actor,
                       ref_policy=model_actor_ref,
                       reward_model=model_critic_ref,
                       value_model=model_critic,
                       train_dataset=dataset['train'],
                       eval_dataset=dataset['test'])
trainer.train()

model_actor.save_pretrained('model/trl')