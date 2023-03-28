#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: finetune-PE
@author: ImKe at 2023/03/25
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""

import os
import sys
import argparse

from typing import List
import fire
import bitsandbytes as bnb
import torch
import torch.nn as nn
from datasets import load_dataset
import transformers
# from transformers import HfArgumentParser, AdapterArguments
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, GPTNeoForCausalLM, AutoTokenizer
from modeling_llama import LlamaForCausalLM
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PrefixTuningConfig,
    TaskType,
)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='alpaca', type=str, required=False, choices=['alpaca', 'alpaca-belle', 'alpaca-belle-cot'])

parser.add_argument("--seed", default=42, type=int, required=False)

parser.add_argument("--load", default=None, type=str, required=False)
parser.add_argument("--train_epochs", default=3, type=int)
parser.add_argument("--learning_rate", default=3e-4, type=float, required=False)
parser.add_argument("--early_stop", default=3, type=int, required=False)
parser.add_argument("--max_length", default=512, type=int, required=False)
parser.add_argument("--val_set_size", default=2000, type=int, required=False)

parser.add_argument("--model_name", default="decapoda-research/llama-7b-hf", type=str, required=False)
parser.add_argument("--pe_name", default="prefix", type=str, required=False,
                    choices=['lora', 'adapter', 'prefix'])
parser.add_argument("--output_dir", default=None, type=str, required=False)

parser.add_argument("--micro_batch_size", default=16, type=int)
parser.add_argument("--batch_size", default=128, type=int)

parser.add_argument("--lora_r", default=8, type=int, required=False)
parser.add_argument("--lora_alpha", default=16, type=int, required=False)
parser.add_argument("--lora_dropout", default=0.05, type=float, required=False)
parser.add_argument("--prefix_num_virtual_token", default=30, type=int, required=False)

args = parser.parse_args()
# optimized for RTX 4090. for larger GPUs, increase some of these?
MICRO_BATCH_SIZE = args.micro_batch_size  # this could actually be 5 but i like powers of 2
BATCH_SIZE = args.batch_size
EPOCHS = args.train_epochs  # we don't always need 3 tbh
LEARNING_RATE = args.learning_rate  # the Karpathy constant
CUTOFF_LEN = args.max_length  # 256 accounts for about 96% of the data
VAL_SET_SIZE = args.val_set_size

RELOAD_FILE_PTH = args.load
DATA_PREFIX = "./data"


if args.dataset == "alpaca":
    DATA_PATH = os.path.join(DATA_PREFIX, "alpaca_data_cleaned.json")
elif args.dataset == "alpaca-belle":
    DATA_PATH = os.path.join(DATA_PREFIX, "alpaca_plus_belle_data.json")
elif args.dataset == "alpaca-belle-cot":
    DATA_PATH = os.path.join(DATA_PREFIX, "alcapa_plus_belle_plus_cot.json")
else:
    print("Error: Wrong type of data.")
    raise NotImplementedError

OUTPUT_DIR = f"./ckpts/saved-{args.dataset}-{args.pe_name}-{args.model_name}" if args.output_dir is None else args.output_dir



def train(
    # model/data params
    base_model: str = MODEL,  # the only required argument
    data_path: str = DATA_PATH,
    output_dir: str = OUTPUT_DIR,
    # training hyperparams
    batch_size: int = BATCH_SIZE,
    micro_batch_size: int = MICRO_BATCH_SIZE,
    num_epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    cutoff_len: int = CUTOFF_LEN,
    val_set_size: int = VAL_SET_SIZE,
    # Parameter-Efficient Component hyperparams
    pe_name: str = args.pe_name,
    lora_r: int = args.lora_r,
    lora_alpha: int = args.lora_alpha,
    lora_dropout: float = args.lora_dropout,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    prefix_num_virtual_token: str = args.prefix_num_virtual_token,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve,
    resume_from_checkpoint: str = RELOAD_FILE_PTH,  # either training checkpoint or final adapter
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    device = "cuda"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = LlamaForCausalLM.from_pretrained(
        os.path.join(CACHE_DIR, base_model),
        load_in_8bit=True,
        device_map=device_map,
    )
    tokenizer = LlamaTokenizer.from_pretrained(os.path.join(CACHE_DIR, base_model))

    model.config.use_cache = True

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    if pe_name == "lora":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
    elif pe_name == "prefix":
        config = PrefixTuningConfig(
            num_virtual_tokens=prefix_num_virtual_token,
            task_type=TaskType.CAUSAL_LM,
        )
    else:
        raise NotImplementedError
    model = get_peft_model(model, config).to(device)

    data = load_dataset("json", data_files=data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = False  # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Response:
{data_point["output"]}"""


if __name__ == "__main__":
    fire.Fire(train)
