import os
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)

from peft import PeftModel
import fire
import evaluate

def longest_common_substring(l1, l2):
    m = len(l1)
    n = len(l2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if l1[i - 1] == l2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_length = max(max_length, dp[i][j])
            else:
                dp[i][j] = 0
    
    return max_length / max(m, n)

def main(
    model_name_or_path="EleutherAI/gpt-j-6B",
    load_lora=True,
    lora_name_or_path="lora",
    metric_name_or_path="rouge",
    val_file="val.json",
    text_column="input",
    label_column="ref",
    per_device_eval_batch_size=1,
    max_src_len=800,
    max_tgt_len=256,
    num_beams=4,
    output_log=False,
    output_path='output/ewc-lora-6B/gen-metric.txt',
):
        
    accelerator = Accelerator()

    eval_dataset = load_dataset(val_file.split(".")[-1], data_files={'validation': val_file})['validation']

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):

        batch = tokenizer(
            examples[text_column],
            max_length=max_src_len,
            padding='max_length',
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt',
        )

        return batch

    with accelerator.main_process_first():
        eval_processed_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()

    def data_collator_longest_padding(features):
        batch = default_data_collator(features)
        max_len = batch['attention_mask'].sum(1).max()
        for k, v in batch.items():
            batch[k] = v[:, -max_len:]
        batch['input_len'] = max_len
        return batch

    eval_dataloader = DataLoader(
        eval_processed_dataset, collate_fn=data_collator_longest_padding, batch_size=per_device_eval_batch_size, pin_memory=True
    )

    #print(next(iter(train_dataloader)))

    # creating model
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_8bit=True, device_map={"": accelerator.local_process_index})
    if load_lora:   
        model = PeftModel.from_pretrained(model, lora_name_or_path, device_map={"": accelerator.local_process_index})
    model.eval()

    gen_kwargs = {
        'max_new_tokens': max_tgt_len, 
        'num_beams': num_beams,
        'pad_token_id': tokenizer.eos_token_id,
    }

    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    accelerator.print(model)

    eval_preds = []
    for _, batch in enumerate(tqdm(eval_dataloader, disable=not accelerator.is_local_main_process)):
        input_len = batch.pop('input_len')
        with torch.no_grad():
            outputs = accelerator.unwrap_model(model).generate(**batch, **gen_kwargs)
        outputs = outputs[:, input_len:]
        outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id).contiguous()
        preds = accelerator.gather_for_metrics(outputs)
        preds = preds.detach().cpu().numpy()
        eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

    assert len(eval_preds) == len(
        eval_dataset[label_column]
    ), f"{len(eval_preds)} != {len(eval_dataset[label_column])}"
    eval_preds = [pred.split('\n')[0].strip() for pred in eval_preds]

    if metric_name_or_path == "rouge":
        eval_scorer = evaluate.load(metric_name_or_path)
        scores = eval_scorer.compute(
            predictions=eval_preds,
            references=eval_dataset[label_column],
            rouge_types=['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        output_str = f"rouge1={scores['rouge1']}, rouge2={scores['rouge2']}, rougeL={scores['rougeL']}"
    elif metric_name_or_path == "lcs":
        lcs_scores = [longest_common_substring(pred.split(', '), ref.split(', ')) for pred, ref in zip(eval_preds, eval_dataset[label_column])]
        lcs_score = sum(lcs_scores) / len(lcs_scores)
        output_str = f'lcs={lcs_score}'
    accelerator.print(f"Evaluation: {output_str}")

    task_name = os.path.basename(val_file).split('.')[0].replace('_', ' ')
    if accelerator.is_local_main_process:
        with open(output_path, "a") as f:
            f.write(f"{task_name}: {output_str}\n")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    fire.Fire(main)
