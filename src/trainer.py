import torch
import requests
import os
from src.model import TinyModel

# 1ere etape: gestion des données:

def prepare_data():
    file_path = 'input.txt'
    # Téléchargement automatique de la data d'entrainement si elle n'existe pas
    if not os.path.exists(file_path):
        print("📥 Téléchargement du dataset TinyShakespeare...")
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(file_path, 'wb') as f:
            f.write(requests.get(url).content)
            
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # creation du vocabulaire ( tous les differents caracteres)
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"📚 Taille du vocabulaire: {vocab_size} caractères uniques.")

    # mapping de chaque caractere a un entier
    stoi = {ch: i for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]

    # conversion du texte en tenseur, on prends les 50 0000 premiers caracteres pour aller vite
    data =  torch.tensor(encode(text[:500000]), dtype=torch.long)
    return data, vocab_size

data, vocab_size = prepare_data()
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split, batch_size):
    # on prepare un petit batch de donnees pour l'entrainement
    data_src= train_data if split == 'train' else val_data
    block_size = 64 # taille de la sequence
    ix = torch.randint(len(data_src) - block_size, (batch_size,))
    x = torch.stack([data_src[i:i+block_size] for i in ix])
    y = torch.stack([data_src[i+1:i+block_size+1] for i in ix])
    return x, y


# 2eme etape: Fonction d'Évaluation

def evaluate_config(config):
     # le but est de prendre en entree un dictionnaire d'hyperparametres (config)
    # entrainer un modele avec ces hyperparametres et retourner la perte de validation

    #etape 1 extraction des hyperparametres
    try: 
        lr = float(config['lr'])
        batch_size = int(config['batch_size'])
        n_embd = int(config['n_embd'])
        n_layer = int(config['n_layer']) # L'algo choisit la profondeur !
        n_head = 4 # On fixe pour éviter les erreurs de dimension (n_embd % n_head)
        dropout = float(config['dropout'])

        # securité n_embd doit etre multiple de n_head
        if n_embd % n_head != 0:
            n_embd = n_head * (n_embd // n_head)

        #etape 2: creation du modele

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = TinyModel(vocab_size, n_embd, n_head, n_layer , dropout)
        model.to(device)
        
        #etape 3: creation de l'optimiseur
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        #etape 4: boucle d'entrainement
        # pour favoriser le merite au hasard, on fixe un nombre d'iterations plutot haut
        max_iters = 500

        model.train()
        for iter in range(max_iters):
            xb, yb = get_batch('train', batch_size)
            xb, yb = xb.to(device), yb.to(device)

            logits, loss = model(xb, yb)
        
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        #etape 5: evaluation finale sur le jeu de validation
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
        print(f"Erreur config: {config}")
        return 999.0 # Retourne une erreur énorme si la config est cassée

    return val_loss