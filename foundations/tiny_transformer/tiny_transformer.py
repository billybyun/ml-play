import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 0. Config
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

batch_size = 32
block_size = 64       # context length
n_layers = 2
n_heads = 4
d_model = 128
d_ff = 512
dropout = 0.1
max_iters = 2000      # keep small at first; you can increase later
eval_interval = 200


# -----------------------------
# 1. Tiny toy dataset (no download)
# -----------------------------
# You can replace this with any longer text you like
text = """
This is a tiny Transformer language model experiment.
It is trained on this small in-code dataset.
You can later replace this text with a bigger file or WikiText-2.
For now, we just want to see that the model trains and can generate something non-trivial.
"""

# Build character-level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size} characters")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s: str):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)

# Train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split: str):
    """Draw a random batch of sequences from train or val data."""
    source = train_data if split == "train" else val_data
    # make sure we don't go out of bounds
    idx = torch.randint(low=0, high=len(source) - block_size - 1, size=(batch_size,))
    x = torch.stack([source[i : i + block_size] for i in idx])
    y = torch.stack([source[i + 1 : i + 1 + block_size] for i in idx])
    return x.to(device), y.to(device)


# -----------------------------
# 2. Model components
# -----------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.key = nn.Linear(d_model, d_model, bias=False)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Causal mask: prevent attending to future positions
        mask = torch.tril(torch.ones(block_size, block_size))
        # shape (1, 1, T, T) so it can broadcast over batch & heads
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape  # batch, time, channels

        # Linear projections
        k = self.key(x)    # (B, T, C)
        q = self.query(x)
        v = self.value(x)

        # Reshape for multi-head: (B, T, C) -> (B, n_heads, T, head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_heads, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v  # (B, n_heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # back to (B, T, C)

        y = self.proj(y)
        y = self.dropout(y)
        return y


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(n_heads, d_model, block_size, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm residual
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        block_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, block_size, dropout)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie token and output embedding weights
        self.head.weight = self.token_emb.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, "Sequence length exceeds block size"

        # Token + positional embeddings
        tok = self.token_emb(idx)  # (B, T, d_model)
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T, d_model)
        x = tok + pos  # broadcast over batch

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            # Crop to last block_size tokens
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            # Take logits for last time step
            logits_last = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits_last, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# -----------------------------
# 3. Training setup
# -----------------------------
model = TinyTransformerLM(
    vocab_size=vocab_size,
    d_model=d_model,
    n_layers=n_layers,
    n_heads=n_heads,
    d_ff=d_ff,
    block_size=block_size,
    dropout=dropout,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


@torch.no_grad()
def estimate_loss(num_batches: int = 50):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(num_batches):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


# -----------------------------
# 4. Training loop
# -----------------------------
print("Starting training on", device)
for step in range(max_iters + 1):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {step:4d} | train loss {losses['train']:.3f} | val loss {losses['val']:.3f}"
        )

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# -----------------------------
# 5. Sample generation
# -----------------------------
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # start with "zero" token
generated = model.generate(context, max_new_tokens=200)[0].tolist()
print("\n=== Generated text ===\n")
print(decode(generated))
print("\n======================\n")

