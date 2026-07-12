# ==============================================================================
#                                  Main script 
# ==============================================================================
import torch 
from tokenization import build_vocab, encode, decode 
from prepare_data import load_data, prepare_text, get_batch
from model_transformer import MiniGPT
from train_model import train

# ==============================================================================
#                                    Hyperparameters 
# ============================================================================== 

batch_size = 32
block_size = 128 
num_epochs = 10000
eval_iters = 100
eval_interval = 500 
learning_rate = 3e-4
dropout = 0.1

n_embd = 128 
num_heads = 4
num_layers = 4 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
#                                    Training
# ============================================================================== 


file_path = "/Users/mariacherifa/Desktop/MiniGPT_for_SFT/TinyStories-train.txt"
text_used = load_data(file_path)

vocabulary, vocabulary_size, itos, stoi = build_vocab(text_used)
data, train_data, val_data = prepare_text(text_used, stoi)

model = MiniGPT(n_embd, block_size, vocabulary_size, num_heads, num_layers, dropout)

train(model, train_data, val_data, batch_size, block_size, eval_iters, learning_rate, num_epochs, device, eval_interval)

prompt = "Once upon a time:\n"
context = torch.tensor(
    [encode(prompt, stoi)],
    dtype=torch.long,
    device=device
)
generated_ids = model.generate(context, max_new_tokens=300)
generated_text = decode(generated_ids[0].tolist(), itos).replace("<|endoftext|>", "").strip()

print("\nGenerated text:")
print(generated_text)

# =================================================================================
#                                    Saving checkpoints 
# =================================================================================
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": vocabulary_size,
        "stoi": stoi,
        "itos": itos,
        "block_size": block_size,
        "n_embd": n_embd,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dropout": dropout,
    },
    "/Users/mariacherifa/Desktop/MiniGPT_for_SFT/minigpt_pretrained.pt",
)



