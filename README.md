# MiniGPT From Scratch

A GPT-style decoder-only Transformer implemented from scratch in PyTorch and trained on the Tiny Shakespeare dataset for next-token prediction.

This repository extends my previous project, Transformer-from-Scratch, by moving from a toy implementation to a complete language model training pipeline. While the first project focused on understanding each Transformer component individually, this project assembles these components into a GPT-style language model capable of learning from real text and generating coherent Shakespeare-like sentences.

The implementation remains intentionally compact and educational, with every component written from scratch using PyTorch.

# Project Overview

<pre>
Tiny Shakespeare
        │
        ▼
Vocabulary Construction
        │
        ▼
Character Tokenization
        │
        ▼
Training / Validation Split
        │
        ▼
Mini-batch Generation
        │
        ▼
Token Embeddings
        │
        ▼
Positional Embeddings
        │
        ▼
Transformer Blocks
   ├── Multi-Head Attention
   ├── Feed Forward Network
   ├── Layer Normalization
   └── Residual Connections
        │
        ▼
Language Modeling Head
        │
        ▼
Cross-Entropy Loss
        │
        ▼
AdamW Optimization
        │
        ▼
Autoregressive Text Generation
</pre>

#  Repository Structure

<pre>
MiniGPT/
├── data.py          # Loads Tiny Shakespeare, prepares tensors, creates mini-batches
├── input.txt        # Tiny Shakespeare dataset
├── main_script.py   # Main script: hyperparameters, model creation, training, generation
├── model.py         # MiniGPT architecture: attention, blocks, embeddings, generation
├── tokenizer.py     # Vocabulary construction, encoding and decoding
└── train.py         # Training loop and loss evaluation
</pre>


# Features

This implementation includes
- Character-level tokenization
- Token and positional embeddings
- Causal self-attention
- Multi-head attention
- Feed-forward networks
- Residual connections
- Layer normalization
- Dropout regularization
- Training and validation loop
- Temperature-based text generation
- Modular implementation

# Improvements over my previous repository

Compared to my first repository (Transformer-from-Scratch), this implementation introduces several components used in modern GPT-style language models:

| Previous repository              | This repository                 |
| -------------------------------- | ------------------------------- |
| Toy dataset                      | Tiny Shakespeare dataset        |
| Basic Transformer implementation | Complete GPT training pipeline  |
| Simple generation                | Temperature-controlled sampling |
| Single training script           | Modular project structure       |
| No regularization                | Dropout                         |
| Basic evaluation                 | Training and validation losses  |

# Running the Project

To train the model and generate text, simply run:

```bash
python main_script.py
```

The `main_script.py` file orchestrates the entire pipeline:

- Loads the Tiny Shakespeare dataset
- Builds the vocabulary and tokenizer
- Prepares the training and validation data
- Creates the MiniGPT model
- Trains the model
- Evaluates the training and validation losses
- Generates text autoregressively from the trained model




