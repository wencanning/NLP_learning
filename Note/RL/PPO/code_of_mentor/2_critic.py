import torch
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


from datasets import load_dataset, concatenate_datasets

dataset = load_dataset('imdb')
dataset = concatenate_datasets([dataset[i] for i in ['train', 'test']])


def collator(data):
    text = [i['text'] for i in data]
    label = [i['label'] for i in data]

    data = tokenizer(text,
                     padding=True,
                     truncation=True,
                     max_length=50,
                     return_tensors='pt').to(device)

    data['labels'] = torch.FloatTensor(label).to(device)

    return data


loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    drop_last=True,
    collate_fn=collator
)

print(len(loader), next(iter(loader)))


from transformers import AutoModelForSequenceClassification

model_critic = AutoModelForSequenceClassification.from_pretrained(
    'EleutherAI/pythia-160m', num_labels=1).to(device)
model_critic.config.pad_token_id = tokenizer.pad_token_id

print(model_critic.config)


optimizer = torch.optim.Adam(model_critic.parameters(), lr=1e-5)

for epoch in range(3):
    for i, data in enumerate(loader):
        out = model_critic(**data)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 1000 == 0:
            logits = (out.logits > 0.5).squeeze(1).long()
            acc = (logits == data['labels'].long()).sum() / len(data['labels'])
            print(epoch, i, len(loader), out.loss.item(), acc.item())

model_critic.save_pretrained('./model/critic')