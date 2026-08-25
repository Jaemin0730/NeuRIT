import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["FSDP_USE_ORIG_PARAMS"] = "true"
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import torch
import transformers

from trainer_add_grad_mask_load_neuron_dict_Llama3_8B_Instruct import Trainer as Trainer_add_grad_mask

os.environ["WANDB_DISABLED"] = "true"


@dataclass
class SFTConfig:
    model_name_or_path: Optional[str] = field(metadata={"help": "Path to pretrained model checkpoint"})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Huggingface dataset name"})
    train_file_path: Optional[str] = field(default=None, metadata={"help": "Path to train data file/directory"})
    validate_file_path: Optional[str] = field(default=None, metadata={"help": "Path to validation data file/directory"})
    max_length: int = field(default=4096, metadata={"help": "Max length of input"})
    text_key_name: Optional[str] = field(default="content",
                                         metadata={"help": "key to text field name in train and validation file"})
    preprocess_num_workers: int = field(default=8,
                                        metadata={"help": "The number of processes to use for the preprocessing."})


def check_file_exist(path: str):
    if not os.path.exists(path):
        raise ValueError(f"Path: {path} not exists!")


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=preds, references=labels)


def main():
    print("--- 11월 11일 수정된 코드 실행됨 ---")
    transformers.set_seed(1234)
    print("--- 의심구역 1 실행됨 ---")
    parser = transformers.HfArgumentParser((SFTConfig, transformers.TrainingArguments))
    print("--- 의심구역 2 실행됨 ---")
    sft_config, training_args = parser.parse_args_into_dataclasses()
    print("--- 의심구역 3 실행됨 ---")

    print(training_args.report_to)

    # check file existence
    if sft_config.dataset_name is None and sft_config.train_file_path is None:
        print("--- 의심구역 4 실행됨 ---")
        raise ValueError(f"One of --dataset_name or --train_file_path must be set")
    if sft_config.train_file_path:
        print("--- 의심구역 5 실행됨 ---")
        check_file_exist(sft_config.train_file_path)
    if sft_config.validate_file_path:
        print("--- 의심구역 6 실행됨 ---")
        check_file_exist(sft_config.validate_file_path)

    # load model, tokenizer
    print("--- 의심구역 7 실행됨 ---")
    tokenizer = transformers.AutoTokenizer.from_pretrained(sft_config.model_name_or_path, padding_side='right',
                                                           trunction_side="right",
                                                           max_length=sft_config.max_length)
    print("--- 의심구역 8 실행됨 ---")
    tokenizer.pad_token = tokenizer.eos_token
    print("--- 의심구역 9 실행됨 ---")

    model = transformers.LlamaForCausalLM.from_pretrained(sft_config.model_name_or_path)
    print("--- 의심구역 10 실행됨 ---")
    for k, v in model.named_parameters():
        print(k)
        if 'up_proj' in k or 'down_proj' in k:
            v.requires_grad = True
        else:
            v.requires_grad = False
    print("--- 의심구역 11 실행됨 ---")
    if sft_config.dataset_name:
        print("--- 의심구역 12 실행됨 ---")
        ds = datasets.load_dataset(sft_config.dataset_name)
        print("--- 의심구역 13 실행됨 ---")
        train_ds, validation_ds = ds['train'], ds['validation']
        print("--- 의심구역 14 실행됨 ---")
        raw_datasets = datasets.DatasetDict({"train": train_ds, "validation": validation_ds})
    else:
        # Split 20% of train data as validation data
        if not sft_config.validate_file_path:
            print("--- 의심구역 15 실행됨 ---")
            train_ds, validation_ds = datasets.load_dataset('json', data_files=sft_config.train_file_path,
                                                            split=['train[:80%]', 'train[80%:]'])
            print("--- 의심구역 16 실행됨 ---")
            raw_datasets = datasets.DatasetDict({"train": train_ds, "validation": validation_ds})
        else:
            print("--- 의심구역 17 실행됨 ---")
            raw_datasets = datasets.load_dataset("json", data_files={'train': sft_config.train_file_path,
                                                                     'validation': sft_config.validate_file_path})
    print("--- 의심구역 18 실행됨 ---")
    print("--- Supervised 시작 ---")
    def process_supervised(record):
        print("--- 의심구역 19 실행됨 ---")
        input_s = record['input']
        output_s = record['output']

        print("--- 의심구역 20 실행됨 ---")
        tokenized = tokenizer([input_s, output_s], add_special_tokens=False)
        print("--- 의심구역 21 실행됨 ---")
        token_ids = [tok_id for tok_ids in tokenized['input_ids'] for tok_id in tok_ids]
        print("--- 의심구역 22 실행됨 ---")
        attention_mask = [mask for masks in tokenized['attention_mask'] for mask in masks]
        print("--- 의심구역 23 실행됨 ---")
        print(token_ids)
        print("--- 의심구역 24 실행됨 ---")
        print(attention_mask)

        # if token_ids[-1] != tokenizer.eos_token_id:
        #     token_ids += [tokenizer.eos_token_id]
        #     attention_mask += [1]
        print("--- 의심구역 25 실행됨 ---")
        processed_record = {
            "input_ids": token_ids[:sft_config.max_length],
            "attention_mask": attention_mask[:sft_config.max_length],
            "labels": token_ids.copy()[:sft_config.max_length]
        }
        print("--- 의심구역 26 실행됨 ---")
        # ignore input label, label is ignored if value is -100
        processed_record["labels"][:min(len(tokenized["input_ids"][0]), sft_config.max_length)] = [-100] * min(
            len(tokenized["input_ids"][0]), sft_config.max_length)
        print("--- 의심구역 27 실행됨 ---")
        return {k: torch.tensor(v, dtype=torch.int) for k, v in processed_record.items()}
    print("--- 의심구역 28 실행됨 ---")
    with training_args.main_process_first(desc="Process supervised dataset"):
        sft_dataset = raw_datasets.map(
            process_supervised,
            batched=False,
            # num_proc=sft_config.preprocess_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            desc="Process supervised dataset"
        )
    print("--- 의심구역 29 실행됨 ---")
    
    
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=sft_dataset["train"],
        eval_dataset=sft_dataset["validation"],
        tokenizer=tokenizer,  # trainer need tokenizer.pad_token_id,
        data_collator=transformers.DataCollatorForTokenClassification(tokenizer=tokenizer, padding="longest",
                                                                      max_length=sft_config.max_length,
                                                                      label_pad_token_id=-100),
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics

    )
    print("--- 의심구역 30 실행됨 ---")
    # trigger Training
    print("--- 모델학습 시작 ---")
    trainer.train()
    print("--- 모델저장 시작 ---")
    trainer.save_model()


if __name__ == '__main__':
    transformers.Trainer = Trainer_add_grad_mask
    main()


