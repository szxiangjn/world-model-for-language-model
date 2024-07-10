import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss


def compute_fisher_matrix(data_path,
         model_name_or_path='EleutherAI/gpt-j-6b', 
         batch_size=1):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto')

    # Get data size
    num = 0
    with open(data_path, 'r') as f:
        for i, line in enumerate(tqdm(f)):
            num += 1

    sampled_indices = np.random.choice(range(num), size=20000, replace=False).tolist()
    sampled_indices = set(sampled_indices)
    samples = []
    print('Loading samples...')
    with open(data_path, 'r') as f:
        for i, line in enumerate(tqdm(f)):
            if i not in sampled_indices:
                continue
            samples.append(json.loads(line))
            if len(samples) == 20000:
                break

    print('Computing fisher matrix...')
    fisher_matrix = defaultdict(float)
    for i, batch in tqdm(enumerate(range(0, len(samples), batch_size))):
        batch_samples = samples[batch:batch+batch_size]
        batch_samples = [sample['text'] for sample in batch_samples]
        inp = tokenizer(batch_samples, return_tensors='pt', max_length=800, truncation=True, padding='longest')

        labels = inp['input_ids'].masked_fill(inp['attention_mask'] == 0, -100)
        logits = model(**inp).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(logits.device)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        loss.backward()
        for n, p in model.named_parameters():
            fisher_matrix[n] = fisher_matrix[n] + (p.grad.data.detach().cpu() * len(batch_samples)) ** 2
        model.zero_grad()
    fisher_matrix = {n: fisher_matrix[n] / len(samples) for n in fisher_matrix}

    return fisher_matrix


