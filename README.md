# AI-Generated Text Detection with DeBERTa-Xlarge

This repository contains code for fine-tuning a DeBERTa-Xlarge model to detect AI-generated text. The model is trained using all active layers and 5 cross-validation approach for performance evaluation.

## Model Overview

- **Model**: DeBERTa-Xlarge
- **Cross-Validation**: 5-fold cross-validation is employed to assess model performance and generalization capability.

## Dataset

The dataset used for fine-tuning the model consists of a combination of several publicly available datasets, as well as generated samples:

1. [Persaude Corpus 2](https://www.kaggle.com/datasets/nbroad/persaude-corpus-2?resource=download)
2. [LLM Detect AI Generated Text Competition Dataset](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470121), where the data lodaers were also referred.
3. [DAIGT V4 Train Dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v4-train-dataset)

In addition, data samples were also generated using **LLaMA** and **GPT-2** models.
