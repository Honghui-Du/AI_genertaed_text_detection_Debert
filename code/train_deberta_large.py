import os
import random
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from transformers import (AutoModelForSequenceClassification, 
                          AutoTokenizer, 
                          get_cosine_schedule_with_warmup, 
                          AdamW)
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
from accelerate import Accelerator
import yaml
# from peft import LoraConfig, get_peft_model, TaskType


from deberta_train.ai_dataset import AiDataset
from deberta_train.ai_loader import AiCollator, AiCollatorTrain



logging.basicConfig(level=logging.INFO)

def preprocess_data(file_path):
    """
    Load and preprocess dataset for binary classification.
    """
    df = pd.read_csv(file_path).rename(columns={"full_text": "text"})
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    return df

def tokenize_dataset(df, tokenizer, max_length):
    """
    Tokenize dataset using the provided tokenizer.
    """
    return tokenizer(
        list(df["text"]),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )


def load_and_preprocess_data(cfg, accelerator):
    """
    Loads and preprocesses data according to the given config.
    Returns the entire dataset and assigns random folds for validation/testing (if applicable).
    """
    data_dir = cfg['input_data_path']
    essay_df = pd.read_csv(data_dir).dropna(subset=["text"]).reset_index(drop=True)

    # Add folds based on the seed, useful for random fold assignment
    rng = random.Random(cfg['seed'])
    essay_df['fold'] = essay_df['text'].apply(lambda x: rng.randint(0, 4))  # Assigns random fold 0-4

    accelerator.print(f"Total data: {essay_df.shape[0]} samples")

    return essay_df


def prepare_dataloaders(cfg, train_df, valid_df, tokenizer, accelerator):
    """
    Prepares DataLoaders using custom dataset and collators.
    """
    dataset_creator = AiDataset(cfg)
    prompt_ids = train_df["prompt_id"].unique().tolist()
    gdf = train_df.groupby("prompt_id")["id"].apply(list).reset_index()
    prompt2ids = dict(zip(gdf["prompt_id"], gdf["id"]))
    # Tokenize and prepare datasets
    train_ds = dataset_creator.get_dataset(train_df)
    valid_ds = dataset_creator.get_dataset(valid_df)

    # Format datasets for easier use
    train_ds.set_format(
        type=None,
        columns=['id', 'input_ids', 'attention_mask', 'generated']
    )

    valid_ds.set_format(
        type=None,
        columns=['id', 'input_ids', 'attention_mask', 'generated']
    )

    kwargs = dict(
        train_ds=train_ds,
        prompt_ids=prompt_ids,
        prompt2ids=prompt2ids,
    )
    # Collators
    data_collator_train = AiCollatorTrain(tokenizer=tokenizer, pad_to_multiple_of=64,kwargs=kwargs)
    data_collator_eval = AiCollator(tokenizer=tokenizer, pad_to_multiple_of=64)

    # DataLoaders
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg['train_params']['per_device_train_batch_size'],
        shuffle=True,
        collate_fn=data_collator_train
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg['train_params']['per_device_eval_batch_size'],
        shuffle=False,
        collate_fn=data_collator_eval
    )

    return train_dl, valid_dl



def evaluate(model, data_loader, accelerator):
    """
    Evaluate the model and return F1 score.
    """
    model.eval()
    preds, truths = [], []

    with torch.no_grad():
        for batch in data_loader:
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=1)
            preds.extend(predictions.cpu().numpy())
            truths.extend(batch["labels"].cpu().numpy())
    
    return f1_score(truths, preds)

def train_model(config):
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="fp16") 
    tokenizer = AutoTokenizer.from_pretrained(config['model']['backbone_path'], use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(config['model']['backbone_path'], num_labels=2)
    ####################################################
    # peft_config = LoraConfig(
    #     task_type=TaskType.SEQ_CLS,  # For sequence classification tasks
    #     r=16,  # Low-rank matrix size
    #     lora_alpha=32,
    #     lora_dropout=0.1,
    #         target_modules=[
    #     "encoder.layer.0.attention.output.dense",
    #     "encoder.layer.0.attention.self.query",
    #     "encoder.layer.0.attention.self.key"]  # ,  # Adjusted for DeBERTa
    # )
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()  # Confirm only LoRA layers are trainabl
    #######################################################################
    df = load_and_preprocess_data(config, accelerator)

    kf = KFold(n_splits=5, shuffle=True, random_state=config['seed'])
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(df)):
        logging.info(f"Starting Fold {fold+1}/5")
        
        train_subset = df.iloc[train_idx].reset_index(drop=True)
        valid_subset = df.iloc[valid_idx].reset_index(drop=True)

        train_dl, valid_dl = prepare_dataloaders(config, train_subset, valid_subset, tokenizer, accelerator)



        optimizer = AdamW(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
        
        num_training_steps = len(train_dl) * config['train_params']['epochs']
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(num_training_steps * 0.1), num_training_steps=num_training_steps)
        
        model, optimizer, train_dl, valid_dl = accelerator.prepare(model, optimizer, train_dl, valid_dl)

        best_f1 = 0
        for epoch in range(config['train_params']['epochs']):
            model.train()
            for batch in tqdm(train_dl, desc=f"Epoch {epoch} [Fold {fold+1}]"):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            f1 = evaluate(model, valid_dl, accelerator)
            logging.info(f"Validation F1 Score for Fold {fold+1}: {f1}")

            if f1 > best_f1:
                best_f1 = f1
                model.save_pretrained(f"{config['outputs']['model_dir']}/fold_{fold+1}")
                tokenizer.save_pretrained(f"{config['outputs']['model_dir']}/fold_{fold+1}")
        
        fold_scores.append(best_f1)
        logging.info(f"Best F1 Score for Fold {fold+1}: {best_f1}")
    
    logging.info(f"Average F1 Score across all folds: {np.mean(fold_scores)}")

if __name__ == "__main__":
    # Load config file
    with open("conf/conf_deberta.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    train_model(config)
