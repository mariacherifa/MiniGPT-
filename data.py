# ========================================================
#                  Data loading and split 
# ========================================================
import torch 
from tokenizer import build_vocab, encode, decode 


# Loading text 

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text 

# Prepare text for batching 

def prepare_text(text, stoi):
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    
    n = int(0.8*len(data))
    train_data = data[:n]
    val_data = data[n:]

    return data, train_data, val_data

# Define batches 

def get_batch(data, batch_size, block_size, device):
    if len(data) <= block_size :
        raise ValueError("data is too short for the chosen block_size")
    
    idx = torch.randint(0, len(data)-block_size, (batch_size,))

    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i+1 : i + block_size+1] for i in idx])

    x = x.to(device)
    y = y.to(device)

    return x, y 
