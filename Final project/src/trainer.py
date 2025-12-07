import torch
import requests
import os
from src.model import TinyModel

# Step 1: Data management

def prepare_data():
    file_path = 'input.txt'
    # Automatic download of training data if it doesn't exist
    if not os.path.exists(file_path):
        print("📥 Downloading TinyShakespeare dataset...")
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(file_path, 'wb') as f:
            f.write(requests.get(url).content)
            
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Create vocabulary (all distinct characters)
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"📚 Vocabulary size: {vocab_size} unique characters.")

    # Mapping from character to integer
    stoi = {ch: i for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]

    # Convert text to tensor; we take the first 500,000 characters to speed things up
    data =  torch.tensor(encode(text[:500000]), dtype=torch.long)
    return data, vocab_size

data, vocab_size = prepare_data()
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split, batch_size):
    # Prepare a small batch of data for training
    data_src= train_data if split == 'train' else val_data
    block_size = 64 # Sequence length
    ix = torch.randint(len(data_src) - block_size, (batch_size,))
    x = torch.stack([data_src[i:i+block_size] for i in ix])
    y = torch.stack([data_src[i+1:i+block_size+1] for i in ix])
    return x, y


# Step 2: Evaluation Function

def evaluate_config(config):
    # The goal is to take a dictionary of hyperparameters (config) as input
    # Train a model with these hyperparameters and return the validation loss

    # Step 1: Extract hyperparameters
    try: 
        lr = float(config['lr'])
        batch_size = int(config['batch_size'])
        n_embd = int(config['n_embd'])
        n_layer = int(config['n_layer']) # The algorithm chooses the depth!
        n_head = 4 # Fixed to avoid dimension errors (n_embd % n_head)
        dropout = float(config['dropout'])

        # Safety: n_embd must be a multiple of n_head
        if n_embd % n_head != 0:
            n_embd = n_head * (n_embd // n_head)

        # Step 2: Create model

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = TinyModel(vocab_size, n_embd, n_head, n_layer , dropout)
        model.to(device)
        
        # Step 3: Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        # Step 4: Training loop
        # To favor merit over luck, we set a relatively high number of iterations
        max_iters = 500

        model.train()
        for iter in range(max_iters):
            xb, yb = get_batch('train', batch_size)
            xb, yb = xb.to(device), yb.to(device)

            logits, loss = model(xb, yb)
        
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Step 5: Final evaluation on the validation set
        model.eval()
        with torch.no_grad():
            losses = []
            for _ in range(20):
                xb, yb = get_batch('val', batch_size)
                xb, yb = xb.to(device), yb.to(device)
                _, val_loss = model(xb, yb)
                losses.append(val_loss.item())
            val_loss = sum(losses) / len(losses)

    except Exception as e:
        print(f"Config error: {config}")
        return 999.0 # Return a huge error if the config is broken

    return val_loss