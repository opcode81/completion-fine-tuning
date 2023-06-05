"""
Fine-Tune SantaCoder on code/text dataset
"""

import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Optional, List

import jsonargparse
import numpy as np
import torch
from datasets import load_dataset
from peft import TaskType
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging as tflogging,
    set_seed,
)
import peft

import fim

log = logging.getLogger(__name__)


@dataclass
class FineTuningConfiguration:
    model_path: str = "bigcode/santacoder"
    dataset_name: str = "bigcode/the-stack-dedup"
    subset: str = "data"
    split: str = "train"
    size_valid_set: int = 4000
    streaming: bool = False
    shuffle_buffer: int = 5000
    data_column: str = "content"
    seq_length: int = 1024
    max_steps: int = 10000
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    eos_token_id: int = 49152
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    num_warmup_steps: int = 100
    weight_decay: float = 0.05
    local_rank: int = 0
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    seed: int = 0
    num_workers: int = None
    output_dir: str = "./checkpoints"
    log_freq: int = 1
    eval_freq: int = 1000
    save_freq: int = 1000
    fim_rate: float = 0
    fim_spm_rate: float = 0
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 8
    lora_target_modules: Optional[List[str]] = None
    lora_dropout = 0.1


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            cfg (FineTuningConfiguration): the configuration
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            fim_rate (float): Rate (0.0 to 1.0) that sample will be permuted with FIM.
            fim_spm_rate (float): Rate (0.0 to 1.0) of FIM permuations that will use SPM.
            seed (int): Seed for random number generator.
    """

    def __init__(
        self,
        cfg: FineTuningConfiguration,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        fim_rate=0.5,
        fim_spm_rate=0.5,
        seed=0,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id else cfg.eos_token_id
        )
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.seed = seed

        (
            self.suffix_tok_id,
            self.prefix_tok_id,
            self.middle_tok_id,
            self.pad_tok_id,
        ) = fim.get_fim_token_ids(self.tokenizer)
        if not self.suffix_tok_id and self.fim_rate > 0:
            print("FIM is not supported by tokenizer, disabling FIM")
            self.fim_rate = 0

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            # obtain buffer where each element contains code as text
            # buffer_len is the total length (in characters) of all buffer elements combined
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break

            # tokenize the buffer elements
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]

            # for each buffer element, either
            #    * use the tokens directly or
            #    * (with some probability) use a transformed version for the fill in the middle (FIM) task
            #      (by selecting two random split points to split the sequence into prefix, middle and suffix,
            #      adding separator tokens in between)
            all_token_ids = []
            np_rng = np.random.RandomState(seed=self.seed)
            for tokenized_input in tokenized_inputs:

                # optionally do FIM permutations
                if self.fim_rate > 0:
                    tokenized_input, np_rng = fim.permute(
                        tokenized_input,
                        np_rng,
                        self.suffix_tok_id,
                        self.prefix_tok_id,
                        self.middle_tok_id,
                        self.pad_tok_id,
                        fim_rate=self.fim_rate,
                        fim_spm_rate=self.fim_spm_rate,
                        truncate_or_pad=False,
                    )
                all_token_ids.extend(tokenized_input + [self.concat_token_id])

            # extract (non-overlapping) subsequences of length seq_length from all_token_ids
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                        "input_ids": torch.LongTensor(example),
                        "labels": torch.LongTensor(example),
                    }


def create_datasets(tokenizer, cfg: FineTuningConfiguration):
    dataset = load_dataset(
        cfg.dataset_name,
        data_dir=cfg.subset,
        split=cfg.split,
        use_auth_token=True,
        num_proc=cfg.num_workers if not cfg.streaming else None,
        streaming=cfg.streaming,
    )
    if cfg.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(cfg.size_valid_set)
        train_data = dataset.skip(cfg.size_valid_set)
        train_data = train_data.shuffle(buffer_size=cfg.shuffle_buffer, seed=cfg.seed)
    else:
        dataset = dataset.train_test_split(test_size=0.005, seed=cfg.seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(
            f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
        )
    chars_per_token = chars_token_ratio(train_data, tokenizer, cfg.data_column)
    log.info(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    train_dataset = ConstantLengthDataset(
        cfg,
        tokenizer,
        train_data,
        infinite=True,
        seq_length=cfg.seq_length,
        chars_per_token=chars_per_token,
        content_field=cfg.data_column,
        fim_rate=cfg.fim_rate,
        fim_spm_rate=cfg.fim_spm_rate,
        seed=cfg.seed,
    )
    valid_dataset = ConstantLengthDataset(
        cfg,
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=cfg.seq_length,
        chars_per_token=chars_per_token,
        content_field=cfg.data_column,
        fim_rate=cfg.fim_rate,
        fim_spm_rate=cfg.fim_spm_rate,
        seed=cfg.seed,
    )

    return train_dataset, valid_dataset


def run_training(cfg: FineTuningConfiguration, train_data, val_data):
    log.info("Loading the model")
    # disable caching mechanism when using gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        trust_remote_code=True,
        use_cache=not cfg.gradient_checkpointing,
    )
    train_data.start_iteration = 0

    run_name = f"santacoder-{cfg.subset}"

    if cfg.use_lora:
        run_name += "-lora"
        peft_cfg = peft.LoraConfig(
            target_modules=cfg.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
        )
        model = peft.get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    log.info(f"Starting main loop")

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=cfg.max_steps,
        eval_steps=cfg.eval_freq,
        save_steps=cfg.save_freq,
        save_total_limit=None,
        logging_steps=cfg.log_freq,
        log_level="debug",
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.num_warmup_steps,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        weight_decay=cfg.weight_decay,
        run_name=run_name,
        report_to=["mlflow"],
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data
    )

    log.info("Training...")
    trainer.train(resume_from_checkpoint=True)

    log.info("Saving last checkpoint of the model")
    model.save_pretrained(os.path.join(cfg.output_dir, "final_checkpoint/"))


def main(cfg: FineTuningConfiguration):
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    tflogging.set_verbosity_info()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_auth_token=True)

    train_dataset, eval_dataset = create_datasets(tokenizer, cfg)

    run_training(cfg, train_dataset, eval_dataset)


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s - %(message)s', stream=sys.stdout,
        level=logging.INFO)
    cfg = jsonargparse.CLI(FineTuningConfiguration, as_positional=False)
    main(cfg)
