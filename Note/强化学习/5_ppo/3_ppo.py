import torch
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
tokenizer.add_special_tokens({'pad_token': '[PAD])'})
tokenizer.padding_side = 'left'

print(tokenizer)

from datasets import load_dataset, concatenate_datasets

dataset = load_dataset('imdb')
dataset = concatenate_datasets(list((dataset.values())))


def collator(data):
    data = [i['text'] for i in data]
    return tokenizer(data,
                     padding=True,
                     truncation=True,
                     max_length=5,
                     return_tensors='pt').input_ids.to(device)


batch_size = 8
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    collate_fn=collator
)

from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from trl.trainer.utils import disable_dropout_in_model

model_actor = AutoModelForCausalLM.from_pretrained('model/actor').to(device)
model_actor_ref = AutoModelForCausalLM.from_pretrained('model/actor').to(
    device)

model_critic = AutoModelForSequenceClassification.from_pretrained(
    'model/critic', num_labels=1).to(device)
model_critic_ref = AutoModelForSequenceClassification.from_pretrained(
    'model/critic', num_labels=1).to(device)

model_actor.generation_config.eos_token_id = None
model_actor.generation_config.pad_token_id = None

for i in [model_actor, model_actor_ref, model_critic, model_critic_ref]:
    disable_dropout_in_model(i)

optimizer = torch.optim.AdamW(
    list(model_actor.parameters()) +
    list(model_critic.parameters()),
    lr=5e-6
)



def get_value(critic, question, answer, shift=True):
    input_ids = torch.cat((question, answer), 1)
    attention_mask = input_ids != tokenizer.pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(input_ids, ~attention_mask, 0)

    #[b, lens, 768]
    last_hidden_state = critic.gpt_neox(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids).last_hidden_state

    #[b, lens]
    value = critic.score(last_hidden_state)

    if shift:
        value = value[:, question.shape[1] - 1:-1].squeeze(-1)

    return value


print(get_value(model_critic,
          torch.randint(100, 10000, [2, 5]).to(device),
          torch.randint(100, 10000, [2, 15]).to(device)).shape)


def get_logprob(actor, question, answer):
    input_ids = torch.cat((question, answer), 1)
    attention_mask = input_ids != tokenizer.pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(input_ids, ~attention_mask, 0)

    logits = actor(input_ids=input_ids,
                   attention_mask=attention_mask,
                   position_ids=position_ids).logits

    logits = logits[:, question.shape[1] - 1:-1]
    logits /= 0.7  # temperature

    logprob = logits.log_softmax(dim=-1)
    logprob = logprob.gather(2, answer.unsqueeze(-1)).squeeze(-1)

    return logprob


print(get_logprob(model_actor,
            torch.randint(100, 10000, [2, 5]).to(device),
            torch.randint(100, 10000, [2, 15]).to(device)).shape)


def get_advantage(value, reward_kl):
    advantage = []
    last = 0
    for i in reversed(range(value.shape[1])):
        value_next = 0.0
        if i < value.shape[1] - 1:
            value_next = value[:, i + 1]

        delta = reward_kl[:, i] + value_next - value[:, i]

        last = delta + 0.95 * last

        advantage.append(last)

    return torch.stack(advantage[::-1], axis=1)


print(get_advantage(torch.randn(4, 25), torch.randn(4, 25)).shape)

from trl.trainer.utils import first_true_indices


@torch.no_grad()
def get_data(question):
    #====answer====
    answer = model_actor.generate(
        input_ids=question,
        attention_mask=(question != tokenizer.pad_token_id).long(),
        min_length=-1,
        max_length=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        top_k=0.0,
        top_p=1.0,
        do_sample=True)

    answer = answer[:, question.shape[1]:]

    #求结束位置
    ends = first_true_indices(answer == tokenizer.pad_token_id).tolist()

    #====prob,value====
    prob_old = get_logprob(model_actor, question, answer)
    prob_ref = get_logprob(model_actor_ref, question, answer)
    value_old = get_value(model_critic, question, answer)
    #这里因为有可能取到最后一个字,所以不能偏移,如果偏移的话,最后一个字的值会被裁剪掉.
    value_ref = get_value(model_critic_ref, question, answer, shift=False)

    #end以后的值value归零
    for i, end in enumerate(ends):
        prob_old[i, end:] = 1.0
        prob_ref[i, end:] = 1.0
        value_old[i, end + 1:] = 0.0

    #====reward====
    reward = []
    for i, end in enumerate(ends):
        #没有eos符号的,置为-1
        if tokenizer.eos_token_id not in answer[i]:
            #reward.append(-1)
            #continue
            pass
        #取最后一个字的value作为reward
        reward.append(value_ref[i, end + question.shape[1] - 1])
    reward = torch.FloatTensor(reward).to(device)

    #====advantage====
    #计算kl散度
    reward_kl = -0.05 * (prob_old - prob_ref)

    #把reward加在最后一个字的kl散度上
    for i, end in enumerate(ends):
        if end == len(answer[i]):
            end = -1
        #assert end == -1

        reward_kl[i, end] += reward[i]

    advantage = get_advantage(value_old, reward_kl)
    returns = advantage + value_old

    #标准化,保持数值稳定
    select = torch.cat([adv[:end] for adv, end in zip(advantage, ends)])
    advantage = (advantage - select.mean()) / (select.var() + 1e-8)**0.5

    #end以后的值归零
    for i, end in enumerate(ends):
        advantage[i, end:] = 0

    return question, answer, ends, prob_old, value_old, advantage, returns

print(get_data(next(iter(loader))))

def train(question, answer, ends, prob_old, value_old, advantage, returns):
    for _ in range(4):
        #重新计算value和prob
        prob_new = get_logprob(model_actor, question, answer)
        value_new = get_value(model_critic, question, answer)

        #end以后的值value归零
        for i, end in enumerate(ends):
            prob_new[i, end:] = 1.0
            value_new[i, end + 1:] = 0

        #计算critic部分的loss
        value_clip = torch.clamp(value_new, value_old - 0.2, value_old + 0.2)
        loss_vf1 = (value_new - returns)**2
        loss_vf2 = (value_clip - returns)**2
        loss_vf = torch.max(loss_vf1, loss_vf2)

        #计算actor部分的loss
        ratio = (prob_new - prob_old).exp()
        loss_pg1 = -advantage * ratio
        loss_pg2 = -advantage * torch.clamp(ratio, 0.8, 1.2)
        loss_pg = torch.max(loss_pg1, loss_pg2)

        #丢弃end之后的部分
        loss_vf = [xi[:end + 1] for xi, end in zip(loss_vf, ends)]
        loss_pg = [xi[:end + 1] for xi, end in zip(loss_pg, ends)]
        loss_vf = torch.cat(loss_vf).mean()
        loss_pg = torch.cat(loss_pg).mean()

        loss = loss_pg + 0.05 * loss_vf
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


print(train(*get_data(next(iter(loader)))))


for i in range(4000):
    train(*get_data(next(iter(loader))))

    if i % 200 == 0:
        print(i)
        input_ids = next(iter(loader))[0:1]

        gen = model_actor.generate(input_ids=input_ids,
                                   min_length=-1,
                                   max_length=50,
                                   pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id,
                                   top_k=0.0,
                                   top_p=1.0,
                                   do_sample=True)

        print(tokenizer.decode(input_ids[0]))
        print('--------')
        print(tokenizer.decode(gen[0, input_ids.shape[1]:]))
        print('====================')

model_actor.save_pretrained('model/ppo')