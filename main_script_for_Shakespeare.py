import torch 
import torch.nn as nn 
import torch.functional as F
from model import Head, MultiHeadAttention, FeedForward, MiniGPT
from data import load_text, prepare_text, get_batch
from tokenizer import build_vocab, encode, decode 
from train import estimate_loss, train

# Hyperparameters  =====================================================

batch_size = 32
block_size = 10 
max_iters = 30000
eval_interval = 100
eval_iters = 100
learning_rate = 1e-3
dropout = 0.2

n_embd = 64 
num_heads = 4
n_layers = 5 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================================

text_used = load_text("/Users/mariacherifa/Desktop/MiniGPT/input.txt")
vocabulary, size_vocabulary, itos, stoi = build_vocab(text_used)
data, train_data, val_data = prepare_text(text_used, stoi)

model = MiniGPT(n_embd, num_heads, size_vocabulary ,block_size, n_layers,dropout).to(device)
train(model, train_data, val_data,learning_rate,eval_interval,batch_size,block_size,
           eval_iters,max_iters,device)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_ids = model.generate(context, max_new_tokens=300)
generated_text = decode(generated_ids[0].tolist(), itos)
print("\nGenerated text:")
print(generated_text)
