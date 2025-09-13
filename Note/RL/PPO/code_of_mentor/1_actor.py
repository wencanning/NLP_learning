import torch
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


from datasets import load_dataset, concatenate_datasets

dataset = load_dataset('imdb')
dataset = concatenate_datasets(list((dataset.values())))
dataset = dataset.remove_columns(['label'])

def collator(data):
    data = [i['text'] for i in data]

    data = tokenizer(data,
                     padding=True,
                     truncation=True,
                     max_length=50,
                     return_tensors='pt').to(device)

    data['labels'] = data['input_ids'].clone()
    select = data['labels'] == tokenizer.pad_token_id
    data['labels'][select] = -100

    return data


loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    drop_last=True,
    collate_fn=collator
)

# test, show the immediate results
# print(len(loader), next(iter(loader)))

# Train Actor with SFT
from transformers import AutoModelForCausalLM

model_actor = AutoModelForCausalLM.from_pretrained(
    'EleutherAI/pythia-160m').to(device)


optimizer = torch.optim.Adam(model_actor.parameters(), lr=1e-5)

for epoch in range(3):
    for i, data in enumerate(loader):
        out = model_actor(**data)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 1000 == 0:
            print(epoch, i, len(loader), out.loss.item())
            prompt = data['input_ids'][0]
            chosen = prompt[5:]
            prompt = prompt[:5]

            gen = model_actor.generate(input_ids=prompt.unsqueeze(0),
                                       min_length=-1,
                                       max_length=32,
                                       pad_token_id=tokenizer.pad_token_id,
                                       eos_token_id=tokenizer.eos_token_id,
                                       top_k=0.0,
                                       top_p=1.0,
                                       do_sample=True)
            gen = gen[0, 5:]

            print('prompt=', tokenizer.decode(prompt))
            print('chosen=', tokenizer.decode(chosen))
            print('gen=', tokenizer.decode(gen))

model_actor.save_pretrained('./model/actor')