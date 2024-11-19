import os
import random
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer, AdamW, 
                          get_cosine_schedule_with_warmup)
from tqdm.auto import tqdm
from accelerate import Accelerator
import pandas as pd
from peft import LoraConfig, TaskType, get_peft_model
import os, sys

from gpt2_train.ai_dataset import AiDataset
from gpt2_train.ai_loader import AiCollator
# Dataset class

    
def prepare_model(cfg, tokenizer):
    base_model = AutoModelForCausalLM.from_pretrained(cfg["model"]["backbone_path"])
    
    peft_config = LoraConfig(
        r=cfg["model"]["lora"]["r"],
        lora_alpha=cfg["model"]["lora"]["lora_alpha"],
        lora_dropout=cfg["model"]["lora"]["lora_dropout"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=cfg["model"]["lora"]["target_modules"],
    )
    
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()  # Ensure only LoRA params are trainable
    model.config.use_cache = False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Or add a new token if needed
        model.resize_token_embeddings(len(tokenizer))  # Resize embeddings if pad_token is newly added


    return model

def train(cfg):
    accelerator = Accelerator(gradient_accumulation_steps=4)

    # Set seed for reproducibility
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
            cfg['model']['backbone_path'],
            use_fast=cfg['model']['tokenizer']['use_fast'],
            padding_side=cfg['model']['tokenizer']['padding_side'],
            truncation_side=cfg['model']['tokenizer']['truncation_side'],
        )
    
    # model = AutoModelForCausalLM.from_pretrained(cfg['model']['backbone_path'])
    # # Check and set pad_token
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token  # Or add a new token if needed
    #     model.resize_token_embeddings(len(tokenizer))  # Resize embeddings if pad_token is newly added


    

    essay_df = pd.read_csv(cfg['input_data_path']).rename(columns={"full_text": "text"})
    essay_df = essay_df.dropna(subset=["text"]).reset_index(drop=True)

    # Split data into train and validation sets
    rng = random.Random(cfg['seed'])
    essay_df['fold'] = essay_df['text'].apply(lambda x: 'train' if rng.random() < 0.98 else 'valid')

    train_df = essay_df[essay_df['fold'] == 'train'].copy().reset_index(drop=True)
    valid_df = essay_df[essay_df['fold'] == 'valid'].copy().reset_index(drop=True)
    with accelerator.main_process_first():
        dataset_creator = AiDataset(cfg)
        train_ds = dataset_creator.get_dataset(train_df)
        valid_ds = dataset_creator.get_dataset(valid_df)
    train_ds.set_format(type=None, columns=['input_ids', 'attention_mask', 'labels'])
    valid_ds.set_format(type=None, columns=['input_ids', 'attention_mask', 'labels'])

    data_collator = AiCollator(tokenizer=tokenizer, pad_to_multiple_of=64)

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['train_params']['per_device_train_batch_size'],
        shuffle=True,
        collate_fn=data_collator,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg['train_params']['per_device_eval_batch_size'],
        shuffle=False,
        collate_fn=data_collator,
    )


    model = prepare_model(cfg, tokenizer)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=cfg['optimizer']['lr'])
    num_training_steps = len(train_loader) * cfg['train_params']['num_train_epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(num_training_steps * cfg['train_params']['warmup_pct']), num_training_steps=num_training_steps
    )


    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader)


    # device = accelerator.device()
    # model.to(device)

    # Training loop
    model.train()
    for epoch in range(cfg['train_params']['num_train_epochs']):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix(loss=loss.item())

        # Validation loop
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                # batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                eval_loss += outputs.loss.item()

        eval_loss /= len(valid_loader)
        print(f"Validation Loss after epoch {epoch}: {eval_loss:.4f}")
        model.train()

    # Save the model
    model.save_pretrained(cfg['outputs']['model_dir'])
    tokenizer.save_pretrained(cfg['outputs']['model_dir'])

if __name__ == "__main__":
    import yaml

    with open("conf/conf_gpt2.yaml", "r") as f:
        config = yaml.safe_load(f)
    train(config)
