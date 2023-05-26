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
import math

def main(
    model_name_or_path="EleutherAI/gpt-j-6B",
    load_lora=True,
    lora_name_or_path="lora",
    val_file="pile_test.json",
    text_column="text",
    per_device_eval_batch_size=1,
    max_src_len=800,
):
    accelerator = Accelerator()

    eval_dataset = load_dataset('json', data_files={'validation': val_file})['validation']
    eval_dataset = eval_dataset.select(range(5000))

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='right')
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
        batch['labels'] = batch['input_ids'].clone()
        batch['labels'][batch['labels'] == tokenizer.pad_token_id] = -100
        batch['seq_len'] = batch['attention_mask'].sum(1) - 1

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
            if k == 'seq_len':
                continue
            batch[k] = v[:, :max_len]
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

    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    accelerator.print(model)

    total_loss = 0
    total_seq_len = 0
    for _, batch in enumerate(tqdm(eval_dataloader, disable=not accelerator.is_local_main_process)):
        seq_len = batch.pop('seq_len').sum()
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss.double()
        loss *= seq_len
        loss = accelerator.gather(loss)
        seq_len = accelerator.gather(seq_len)
        total_loss += loss.sum().cpu().item()
        total_seq_len += seq_len.sum().cpu().item()

    # ppl = math.exp(total_loss / total_seq_len)
    ppl = math.pow(2, total_loss / total_seq_len)
    accelerator.print(f"Evaluation: ppl={ppl}")
        
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    fire.Fire(main)
