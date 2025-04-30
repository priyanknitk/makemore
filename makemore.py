"""
A minimalistic character-level language model using PyTorch.
"""

import random
from pathlib import Path
import torch
import torch.nn.functional as F

# ---------- data ----------------------------------------------------------- #
def load_words(path: str = 'names.txt'):
    return Path(path).read_text().splitlines()

def build_vocab(words):
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos

def build_dataset(words, block_size, stoi):
    X, Y = [], []
    for w in words:
        ctx = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(ctx)
            Y.append(ix)
            ctx = ctx[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

# ---------- minimal NN framework (unchanged) ------------------------------- #
class Linear:

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in** 0.5
        self.bias = torch.zeros(fan_out) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])
    
class BatchNorm1d:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        if self.training:
            # compute mean and variance from current batch
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                # for 3D input, we need to compute mean and variance over the last dimension
                dim = (0, 1)
            xmean = x.mean(dim=dim, keepdim=True)
            xvar = x.var(dim=dim, keepdim=True, unbiased=True)
        else:
            # use running mean and variance for inference
            xmean = self.running_mean
            xvar = self.running_var
        
        # normalize the input
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:

    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []
    
class Embedding:

    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
    
    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out
    
    def parameters(self):
        return [self.weight]
    
class FlattenConsecutive:

    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out
    
    def parameters(self):
        return []

class Sequential:
    
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def zero_grad(self):
        for layer in self.layers:
            for p in layer.parameters():
                p.grad = None

# ---------- model / training / sampling ------------------------------------ #
def create_model(vocab_size, n_embed=24, n_hidden=128):
    model = Sequential([
        Embedding(vocab_size, n_embed),
        FlattenConsecutive(2), Linear(n_embed * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, vocab_size),
    ])
    with torch.no_grad():
        model.layers[-1].weight *= 0.1
    return model

def train(model, Xtr, Ytr, max_steps=200_000, batch_size=32):
    params = model.parameters()
    for p in params: p.requires_grad = True
    for i in range(max_steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
        Xb, Yb = Xtr[ix], Ytr[ix]
        logits = model(Xb)
        loss = F.cross_entropy(logits, Yb)
        model.zero_grad()
        loss.backward()
        lr = 0.1 if i < 100_000 else 0.01
        for p in params:
            p.data -= lr * p.grad
        if i % 10_000 == 0:
            print(f'{i:7d}/{max_steps}: {loss.item():.4f}')

@torch.no_grad()
def split_loss(model, x, y, tag):
    print(f'{tag} loss: {F.cross_entropy(model(x), y).item():.4f}')

@torch.no_grad()
def sample(model, block_size, itos, n=20, seed=2147483657):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        out, ctx = [], [0] * block_size
        while True:
            probs = F.softmax(model(torch.tensor([ctx])), dim=1)
            ix = torch.multinomial(probs, 1, generator=g).item()
            ctx = ctx[1:] + [ix]
            if ix == 0: break
            out.append(ix)
        print(''.join(itos[i] for i in out))

# ---------- command‑line entry point --------------------------------------- #
def main():
    random.seed(42); torch.manual_seed(42)
    words = load_words()
    stoi, itos = build_vocab(words)
    vocab_size, block_size = len(itos), 8

    random.shuffle(words)
    n1, n2 = int(.8*len(words)), int(.9*len(words))
    Xtr, Ytr = build_dataset(words[:n1], block_size, stoi)
    Xdev, Ydev = build_dataset(words[n1:n2], block_size, stoi)
    Xte, Yte  = build_dataset(words[n2:] , block_size, stoi)

    model = create_model(vocab_size)
    train(model, Xtr, Ytr)

    for layer in model.layers:
        layer.training = False  # set to eval mode

    split_loss(model, Xtr, Ytr, 'train')
    split_loss(model, Xdev, Ydev, 'val')
    split_loss(model, Xte,  Yte,  'test')
    sample(model, block_size, itos)

if __name__ == '__main__':
    main()