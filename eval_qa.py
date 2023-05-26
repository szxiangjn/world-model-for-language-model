import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm
import numpy as np
from peft import PeftModel

norm_dict = {
    'housework_qa': 'un',
    'neg_housework_qa': 'un',
    'act_infer': 'un',
    'act_recog': 'ln',
    'count': 'ln',
    'obj_move': 'ln'
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument('--model_name_or_path', type=str, default='EleutherAI/gpt-j-6B')
    parser.add_argument('--load_lora', type=int, default=1)
    parser.add_argument('--lora_name_or_path', type=str, default=None)
    parser.add_argument('--output_log', action="store_true")
    parser.add_argument('--output_path', type=str, default='output/ewc-lora-6B/qa-metric.txt')
    args = parser.parse_args()
    return args

def main(args):
    norm = norm_dict[args.task_name]
    acc_list = []

    # Prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, load_in_8bit=True, device_map={"": 0})
    if args.load_lora:
        model = PeftModel.from_pretrained(model, args.lora_name_or_path, device_map={"": 0})
    model.eval()

    def compute_prob(inp, contxt_len, answer_tokens):
        inputs = tokenizer(inp, return_tensors='pt')
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        logits = logits[:, contxt_len - 1:inputs['attention_mask'].sum()]
        vocab_log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            vocab_log_probs, dim=2, index=answer_tokens[:, :, None]
        )
        log_prob = token_log_probs.sum()
        return log_prob.cpu().item()

    answer_contxt_len = tokenizer('Answer:', return_tensors="pt").input_ids.size(1)
    with torch.no_grad():
        with open(args.data_path, "r") as f:
            for line in tqdm(list(f)):
                sample = json.loads(line)
                prompt = sample["input"]
                prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.size(1)
                prob_list = [] # list of log prob of each answer
                answer_list = sample['answer_list']

                for answer in answer_list:
                    answer_tokens = tokenizer(f' {answer}', return_tensors='pt').input_ids.to(model.device)
                    if norm == 'ln':
                        prob = compute_prob(f'{prompt} {answer}', prompt_len, answer_tokens)
                        final_prob = prob / answer_tokens.size(1)
                    elif norm == 'un':
                        prob = compute_prob(f'{prompt} {answer}', prompt_len, answer_tokens)
                        uncond_prob = compute_prob(f'Answer: {answer}', answer_contxt_len, answer_tokens)
                        final_prob = prob - uncond_prob
                    else:
                        raise NotImplementedError
                    prob_list.append(final_prob)
                    
                gt_idx = answer_list.index(sample['answer'])
                gen_idx = np.argmax(prob_list)
                acc_list.append(gt_idx == gen_idx)
            
    acc = sum(acc_list) / len(acc_list)
    return acc

if __name__ == "__main__":
    args = parse_args()
    args.task_name = os.path.basename(args.data_path).split('.')[0]
    acc = main(args)

    task_name = args.task_name.replace('_', ' ')
    output_str = f"{task_name}: {acc:.4f}"
    print(output_str)
    if args.output_log:
        with open(args.output_path, 'a') as f:
            f.write(output_str + '\n')
