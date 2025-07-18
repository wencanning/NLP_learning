import torch
import random
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = 'left'

print(tokenizer)

from datasets import load_dataset, concatenate_datasets

dataset = load_dataset('imdb')
dataset = concatenate_datasets(list(dataset.values()))
dataset = dataset.remove_columns(['label'])

print(dataset, dataset[0])


from transformers import AutoModelForCausalLM

# load model from manual ppo
model_actor = AutoModelForCausalLM.from_pretrained('model/ppo').to(device)
# load model from trl ppo
# model_actor = AutoModelForCausalLM.from_pretrained('model/trl').to(device)

print(model_actor.config)

#====question====
question = random.choices(dataset, k=12)
question = [i['text'] for i in question]

question = tokenizer(question,
                     padding=True,
                     truncation=True,
                     max_length=5,
                     return_tensors='pt').input_ids.to(device)

#====answer====
answer = model_actor.generate(input_ids=question,
                              min_length=-1,
                              max_length=50,
                              pad_token_id=tokenizer.pad_token_id,
                              eos_token_id=tokenizer.eos_token_id,
                              top_k=0.0,
                              top_p=1.0,
                              do_sample=True)
answer = answer[:, question.shape[1]:]

for q, a in zip(question, answer):
    print(tokenizer.decode(q), '->', tokenizer.decode(a))
    print('==============')
