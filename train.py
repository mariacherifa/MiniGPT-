# ========================================================
#                     Training MiniGPT
# ========================================================

import torch 
import torch.nn as nn
from model import Head, MultiHeadAttention, Block, MiniGPT
from data import load_text, prepare_text, get_batch
from tokenizer import build_vocab, encode, decode


# Estimate loss in training and evaluation 

@torch.no_grad()

def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, device):
    result = {}

    model.eval()

    for split, data in [("train", train_data), ("val", val_data)]:

        losses =[]

        for _ in range(eval_iters):
            x,y = get_batch(data, batch_size, block_size, device)
            logits, loss = model(x,y)
            losses.append(loss.item())

        result[split] = sum(losses) / len(losses)

    model.train()

    return result 

# Training 

def train(model, train_data, val_data,learning_rate,eval_interval,batch_size,block_size,eval_iters,max_iters,device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iters):
        if step % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data,
                                    batch_size, block_size, eval_iters, 
                                    device)
            print(
                f"step {step}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}")
            
        x, y = get_batch(train_data, batch_size, block_size, device)

        logits, loss = model(x,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model 

    

