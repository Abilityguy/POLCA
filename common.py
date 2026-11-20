"""Shared pieces: the model, the addition data, the loss, token labels and checkpoint helpers.

The model is the paper's 3-layer Transformer (9,475,594 parameters) written in plain PyTorch.
Data generator, loss and optimiser settings follow the authors' released
train_arithmetic_model.py (github.com/skangasl/POLCA).
"""

import json
import math

import torch
import torch.nn.functional as F
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_DIGITS = 3
PLUS_TOKEN, EQUALS_TOKEN = 10, 11
SEQ_LEN = 3 * NUM_DIGITS + 3  # "AAA+BBB=CCCC"
PLACES = ["1000s", "100s", "10s", "1s"]  # answer order

MODEL_CFG = dict(n_layers=3, d_model=512, n_heads=4, d_head=128, d_mlp=2048, d_vocab=12, d_vocab_out=10, n_ctx=SEQ_LEN)
BATCH_SIZE = 64
TEST_SIZE = 1250  # problems in the test set and in each Hessian sample (5,000 answer tokens)
LR, BETAS, WEIGHT_DECAY, WARMUP = 1e-5, (0.9, 0.95), 0.1, 100


# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, d_head):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        self.W_Q = nn.Linear(d_model, n_heads * d_head)
        self.W_K = nn.Linear(d_model, n_heads * d_head)
        self.W_V = nn.Linear(d_model, n_heads * d_head)
        self.W_O = nn.Linear(n_heads * d_head, d_model)

    def forward(self, x):
        b, t, _ = x.shape

        def split(z):
            return z.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split(self.W_Q(x)), split(self.W_K(x)), split(self.W_V(x))
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        pattern = scores.masked_fill(causal, float("-inf")).softmax(dim=-1)
        z = (pattern @ v).transpose(1, 2).reshape(b, t, self.n_heads * self.d_head)
        return self.W_O(z)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = Attention(d_model, n_heads, d_head)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.W_in = nn.Linear(d_model, d_mlp)
        self.W_out = nn.Linear(d_mlp, d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.W_out(F.relu(self.W_in(self.ln2(x))))


class ArithmeticTransformer(nn.Module):
    """Decoder-only Transformer of the released code (TransformerLens config, pre-LayerNorm, ReLU MLP).

    Initialisation as TransformerLens' default: weights ~ N(0, 0.8 / sqrt(d_model)), biases 0,
    LayerNorm weight 1 and bias 0.
    """

    def __init__(self, n_layers, d_model, n_heads, d_head, d_mlp, d_vocab, d_vocab_out, n_ctx):
        super().__init__()
        self.W_E = nn.Embedding(d_vocab, d_model)
        self.W_pos = nn.Parameter(torch.empty(n_ctx, d_model))
        self.blocks = nn.ModuleList(Block(d_model, n_heads, d_head, d_mlp) for _ in range(n_layers))
        self.ln_final = nn.LayerNorm(d_model, eps=1e-5)
        self.W_U = nn.Linear(d_model, d_vocab_out)
        std = 0.8 / math.sqrt(d_model)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=std)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.W_E.weight, std=std)
        nn.init.normal_(self.W_pos, std=std)

    def forward(self, tokens):
        x = self.W_E(tokens) + self.W_pos[: tokens.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.W_U(self.ln_final(x))


def make_model():
    return ArithmeticTransformer(**MODEL_CFG)


# ---------------------------------------------------------------------------------------------
# Data and loss (released code)
# ---------------------------------------------------------------------------------------------


def data_generator(batch_size, seed):
    """Endless batches of 3-digit additions: (tokens (B, 12), carries (B, 4)).

    As in the released code, the global torch RNG is reseeded with `seed` on the first draw.
    """
    torch.manual_seed(seed)
    while True:
        batch = torch.zeros((batch_size, SEQ_LEN), dtype=torch.int64)
        x = torch.randint(0, 10, (batch_size, NUM_DIGITS))
        y = torch.randint(0, 10, (batch_size, NUM_DIGITS))
        batch[:, :NUM_DIGITS] = x
        batch[:, NUM_DIGITS] = PLUS_TOKEN
        batch[:, 1 + NUM_DIGITS : 1 + 2 * NUM_DIGITS] = y
        batch[:, 1 + 2 * NUM_DIGITS] = EQUALS_TOKEN
        carries = [torch.zeros((batch_size,), dtype=torch.int64)]
        for i in range(NUM_DIGITS):
            digit_sum = batch[:, NUM_DIGITS - 1 - i] + batch[:, 2 * NUM_DIGITS - i] + carries[-1]
            batch[:, -1 - i] = digit_sum % 10
            carries.append((digit_sum >= 10).to(torch.int64))
        batch[:, -1 - NUM_DIGITS] = carries[-1]
        yield batch.to(device), torch.stack(carries, dim=1).to(device)


def loss_fn(logits, tokens, per_token=False):
    """Cross-entropy of the 4 answer tokens (float64). per_token=True returns (examples, 4)."""
    log_probs = F.log_softmax(logits[:, -(NUM_DIGITS + 2) : -1].to(torch.float64), dim=-1)
    token_loss = -torch.gather(log_probs, -1, tokens[:, -(NUM_DIGITS + 1) :, None])[..., 0]
    return token_loss if per_token else token_loss.mean()


def token_labels(tokens):
    """Labels for the 4 answer tokens of each problem, each (examples, 4), in answer order.

    place: 0 = 1000s, 1 = 100s, 2 = 10s, 3 = 1s
    carry: 1 if a carry flows into that place from the place to its right (never for the 1s)
    Flatten with .reshape(-1) for the token order used everywhere: problem * 4 + place.
    """
    tokens = torch.as_tensor(tokens).cpu()
    a, b = tokens[:, :NUM_DIGITS], tokens[:, NUM_DIGITS + 1 : 2 * NUM_DIGITS + 1]
    n = tokens.shape[0]
    carry_into = torch.zeros(n, NUM_DIGITS + 1, dtype=torch.int64)  # indexed from the right
    for i in range(NUM_DIGITS):
        s = a[:, NUM_DIGITS - 1 - i] + b[:, NUM_DIGITS - 1 - i] + carry_into[:, i]
        carry_into[:, i + 1] = (s >= 10).to(torch.int64)
    place = torch.arange(NUM_DIGITS + 1).expand(n, -1)
    return place.numpy().copy(), carry_into.flip(1).numpy()


# ---------------------------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------------------------


def checkpoint_path(run_dir, step):
    """Weights after the optimiser update of loop iteration `step` (step + 1 updates)."""
    return f"{run_dir}/checkpoints/model_step_{step}.pt"


def save_checkpoint(model, path):
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)


def load_checkpoint(path):
    """Build the model and load saved weights.

    Building the model draws its random initial weights first, which advances the global torch
    RNG; basis.py relies on this to reproduce the released pipeline's random numbers.
    """
    model = make_model()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)


def load_config(run_dir):
    with open(f"{run_dir}/config.json") as f:
        return json.load(f)
