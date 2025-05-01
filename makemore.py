"""
A minimalistic character‑level language model using PyTorch.
"""
# pylint: disable=invalid-name, too-many-arguments, too-many-locals

import random
from pathlib import Path
import torch
import torch.nn.functional as F


# ---------- data ----------------------------------------------------------- #
def load_words(path: str = 'names.txt'):
    """Return a list with one word per line from *path*."""
    return Path(path).read_text(encoding='utf-8').splitlines()


def build_vocab(words):
    """Build and return stoi / itos dictionaries for the given *words*."""
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def build_dataset(words, block_size, stoi):
    """Vectorise *words* into context/target tensors."""
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
    """A simple fully‑connected layer implemented with raw tensors."""

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None
        self.out = None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        """Return a list with the weight and bias tensors."""
        return [self.weight] + ([self.bias] if self.bias is not None else [])


class BatchNorm1d:
    """Batch‑normalisation that works for 2‑D and 3‑D inputs."""

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
        self.out = None

    def __call__(self, x):
        if self.training:
            # compute mean and variance from current batch
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                # for 3D input, we need to compute mean and variance
                # over the last dimension
                dim = (0, 1)
            else:
                raise ValueError(f"Unsupported input dimension: {x.ndim}")
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
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * xmean
                )
                self.running_var = (
                    (1 - self.momentum) * self.running_var
                    + self.momentum * xvar
                )
        return self.out

    def parameters(self):
        """Return a list with the gamma and beta tensors."""
        return [self.gamma, self.beta]


class Tanh:
    """Plain Tanh activation with a parameters() stub."""

    def __init__(self):
        self.out = None

    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        """Return an empty list of parameters."""
        return []


class Embedding:
    """Lightweight embedding layer."""

    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
        self.out = None

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        """Return a list with the weight tensor."""
        return [self.weight]


class FlattenConsecutive:
    """Flatten *n* consecutive time steps into the channel dimension."""

    def __init__(self, n):
        self.n = n
        self.out = None

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        """Return an empty list of parameters."""
        return []


class Sequential:
    """Minimal replacement for torch.nn.Sequential."""

    def __init__(self, layers):
        self.layers = layers
        self.out = None

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        """Return a flat list of all parameters in the model."""
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        """Set all gradients to zero."""
        for layer in self.layers:
            for p in layer.parameters():
                p.grad = None


# ---------- model / training / sampling ------------------------------------ #
def create_model(vocab_size, n_embed=24, n_hidden=128):
    """Construct and return the makemore model."""
    # pylint: disable=line-too-long
    model = Sequential([
        Embedding(vocab_size, n_embed), FlattenConsecutive(2),
        Linear(n_embed * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(), FlattenConsecutive(2),  # noqa: E501
        Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(), FlattenConsecutive(2),  # noqa: E501
        Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),  # noqa: E501
        Linear(n_hidden, vocab_size),
    ])
    # pylint: enable=line-too-long
    with torch.no_grad():
        model.layers[-1].weight *= 0.1
    return model


def train(model, x_train, y_train, max_steps=200_000, batch_size=32):
    """Manual SGD loop with a simple learning‑rate schedule."""
    params = model.parameters()
    for p in params:
        p.requires_grad = True
    for step in range(max_steps):
        idx = torch.randint(0, x_train.shape[0], (batch_size,))
        x_batch, y_batch = x_train[idx], y_train[idx]
        logits = model(x_batch)
        loss = F.cross_entropy(logits, y_batch)
        model.zero_grad()
        loss.backward()
        lr = 0.1 if step < 100_000 else 0.01
        for p in params:
            p.data -= lr * p.grad
        if step % 10_000 == 0:
            print(f'{step:7d}/{max_steps}: {loss.item():.4f}')


@torch.no_grad()
def split_loss(model, x_data, y_data, tag):
    """Print cross‑entropy loss for a given split."""
    print(f'{tag} loss: {F.cross_entropy(model(x_data), y_data).item():.4f}')


@torch.no_grad()
def sample(model, block_size, itos, n=20, seed=2147483657):
    """Sample *n* words from the model, starting with a random seed."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        out, ctx = [], [0] * block_size
        while True:
            probs = F.softmax(model(torch.tensor([ctx])), dim=1)
            ix = torch.multinomial(probs, 1, generator=g).item()
            ctx = ctx[1:] + [ix]
            if ix == 0:
                break
            out.append(ix)
        print(''.join(itos[i] for i in out))

# ---------- command‑line entry point --------------------------------------- #


def main():
    """Script entry‑point: prepare data, train the model and sample."""
    random.seed(42)
    torch.manual_seed(42)
    words = load_words()
    stoi, itos = build_vocab(words)
    vocab_size, block_size = len(itos), 8

    random.shuffle(words)
    n1, n2 = int(.8 * len(words)), int(.9 * len(words))
    x_train, y_train = build_dataset(words[:n1], block_size, stoi)
    x_dev, y_dev = build_dataset(words[n1:n2], block_size, stoi)
    x_test, y_test = build_dataset(words[n2:],  block_size, stoi)

    model = create_model(vocab_size)
    train(model, x_train, y_train)

    for layer in model.layers:
        layer.training = False  # evaluation mode

    split_loss(model, x_train, y_train, 'train')
    split_loss(model, x_dev,  y_dev,  'val')
    split_loss(model, x_test, y_test, 'test')
    sample(model, block_size, itos)


if __name__ == '__main__':
    main()
