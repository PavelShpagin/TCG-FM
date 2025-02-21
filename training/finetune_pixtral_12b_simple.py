#!/usr/bin/env python
import os
import json
import math
import yaml
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import torch
import pandas as pd
from numba import cuda
from torch.utils.data import DataLoader
import torch.optim as optim

from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    get_scheduler,
    BitsAndBytesConfig
)
from peft import LoraConfig, TaskType, get_peft_model
from accelerate import Accelerator
import wandb
from huggingface_hub import notebook_login
from dotenv import load_dotenv, dotenv_values
from trl import SFTConfig, SFTTrainer

# Import prompt definitions from prompts.py
from prompts import AUTOFILL_PROMPT

# Import helper functions
from pipeline_utils import collate_fn

@dataclass
class ModelConfig:
    name: str
    repo_id: str
    output_dir: str

@dataclass 
class TrainingConfig:
    num_train_epochs: int
    per_device_batch_size: int
    learning_rate: float
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    gradient_checkpointing_steps: int
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    fp16: bool
    tf32: bool

@dataclass
class PeftLoraConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    bias: str
    task_type: TaskType

@dataclass
class LoggingConfig:
    logging_steps: int
    save_steps: int
    project_name: str
    run_name: str

@dataclass
class DataConfig:
    train_file: str

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    lora: PeftLoraConfig
    logging: LoggingConfig
    data: DataConfig

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            lora=PeftLoraConfig(**config_dict['lora']),
            logging=LoggingConfig(**config_dict['logging']),
            data=DataConfig(**config_dict['data'])
        )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config.from_yaml(args.config)
    
    # Load environment variables
    load_dotenv()

    secrets = dotenv_values(".env")
    # Authenticate with Hugging Face and wandb
    wandb.login(key=secrets.get('WANDB_API_KEY'))
    
    # Initialize wandb run
    wandb.init(
        project=config.logging.project_name,
        name=config.logging.run_name,
        config=asdict(config)
    )
    
    # Load dataset
    if not os.path.exists(config.data.train_file):
        raise FileNotFoundError(f"Dataset not found at {config.data.train_file}")
        
    training_dataset = []
    with open(config.data.train_file, "r") as f:
        for line in f:
            training_dataset.append(json.loads(line.strip()))

    # Check CUDA compatibility
    try:
        cuda.select_device(0)
        cuda.close()
    except Exception as e:
        print("CUDA device selection error:", e)

    # Setup Model and Processor
    processor = AutoProcessor.from_pretrained(config.model.name)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        target_modules=config.lora.target_modules,
        task_type=config.lora.task_type,
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        config.model.name,
        low_cpu_mem_usage=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    
    # Define training arguments using SFTConfig
    training_args = SFTConfig(
        output_dir=config.model.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        gradient_checkpointing=config.training.gradient_checkpointing,
        optim="adamw_torch_fused",
        lr_scheduler_type="constant",
        save_steps=config.logging.save_steps,
        warmup_ratio=config.training.warmup_ratio,
        logging_steps=config.logging.logging_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        bf16=not config.training.fp16,
        tf32=config.training.tf32,
        use_liger=True,
        push_to_hub=True,
        report_to="wandb",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    training_args.remove_unused_columns = False

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=training_dataset,
        peft_config=lora_config,
        data_collator=lambda examples: collate_fn(examples, processor)
    )

    # Start training
    trainer.train()

    # Save the model
    trainer.save_model(training_args.output_dir)
    print("Training complete.")

if __name__ == "__main__":
    main()
