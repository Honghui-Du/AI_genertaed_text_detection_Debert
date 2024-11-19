import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from accelerate import Accelerator
import pandas as pd
from tqdm.auto import tqdm
import yaml
import sys
import random


def get_instruction(inputs):
    ret = f"""
        Prompt: {inputs['prompt_name']}
        Task: {inputs['task']}
        Score: {inputs['holistic_essay_score']}
        Student Grade Level: {inputs['grade_level']}
        English Language Learner: {inputs['ell_status']}
        Disability Status: {inputs['student_disability_status']}
        """.strip()
    n_chars = random.randint(16, 64)

    start = inputs['text'][:n_chars]

    ret = f"### Instruction:\n{ret}\n\n### Response: {start}"
    return ret

def load_model(cfg, accelerator):
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model_path"])
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model_path"])
    
    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))


    model = PeftModel.from_pretrained(model, cfg["adapter_path"])
    model = accelerator.prepare(model)
    model.eval()
    return model, tokenizer

def pre_process_essay(essay_df):

    essay_df = essay_df[~essay_df['text'].isna()].copy()
    essay_df = essay_df.reset_index(drop=True)

    essay_df["student_disability_status"] = essay_df["student_disability_status"].fillna("Unknown")
    essay_df["ell_status"] = essay_df["ell_status"].fillna("Unknown")
    essay_df["grade_level"] = essay_df["grade_level"].fillna(-1)
    essay_df["holistic_essay_score"] = essay_df["holistic_essay_score"].fillna(-1)

    essay_df["prompt"] = essay_df.apply(get_instruction, axis=1)
    return essay_df



def generate_text(cfg):
    accelerator = Accelerator()
    model, tokenizer = load_model(cfg, accelerator)
    
    # Load prompts
    essay_df = pd.read_csv(cfg["input_data_path"]).rename(columns={"full_text": "text"})
    essay_df = pre_process_essay(essay_df)
    prompts = essay_df["prompt"].values.tolist()
    
    os.makedirs(cfg['output_dir'], exist_ok=True)
    
    progress_bar = tqdm(prompts, desc="Generating")
    for i, prompt in enumerate(progress_bar):
        # inputs = get_inputs(prompt, tokenizer, n=1)
        # max_input_length = 2048  - cfg['max_new_tokens']
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(accelerator.device)

        max_position_embeddings = model.config.max_position_embeddings
        input_length = inputs['input_ids'].shape[1]
        max_new_tokens_allowed = max_position_embeddings - input_length

        # inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens_allowed,
                temperature=cfg.get('temperature', 1.0),
                top_k=50
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Save to file
        with open(f"{cfg['output_dir']}/generated_{i}.txt", "w") as f:
            f.write(generated_text)

        progress_bar.set_postfix(prompt=prompt[:20], gen=generated_text[:20])
    print("Text generation completed.")

if __name__ == "__main__":
    with open("conf/conf_generate_gpt2.yaml", "r") as f:
        config = yaml.safe_load(f)
    generate_text(config)
