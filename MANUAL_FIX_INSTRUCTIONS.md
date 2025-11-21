# Step-by-Step Manual Instructions to Fix the Model

## Overview
Your current model uses GPT-style attention (combined c_attn/c_proj), but the reference model uses LLaMA-style attention with separate q/k/v/o projections and Grouped Query Attention (GQA).

## Changes Required

### Change 1: Update LlamaAttention Class (Cell 5)

**Step 1.1:** Find Cell 5 which contains `class LlamaAttention`

**Step 1.2:** Replace the `__init__` method. Find this code:
```python
def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
```

**Replace with:**
```python
def __init__(self, config):
    super().__init__()
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.num_key_value_heads = config.num_key_value_heads
    self.head_dim = config.n_embd // config.n_head
    
    # Separate projections for q, k, v, o (LLaMA style)
    # q_proj: all heads, k_proj and v_proj: only num_key_value_heads (GQA)
    self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
    self.k_proj = nn.Linear(config.n_embd, config.num_key_value_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(config.n_embd, config.num_key_value_heads * self.head_dim, bias=False)
    self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
    
    # Rotary positional embedding (RoPE) - applied to q and k in forward
    self.rotary_emb = RotaryEmbedding(self.head_dim)
    
    # Causal mask
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
```

**Step 1.3:** Replace the `forward` method. Find this code:
```python
def forward(self, x):
    B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
    # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh,hs)
    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

    y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    # output projection
    y = self.c_proj(y)
    return y
```

**Replace with:**
```python
def forward(self, x):
    B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
    
    # Project to q, k, v
    q = self.q_proj(x)  # (B, T, n_embd)
    k = self.k_proj(x)  # (B, T, num_key_value_heads * head_dim)
    v = self.v_proj(x)  # (B, T, num_key_value_heads * head_dim)
    
    # Reshape for multi-head attention
    q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
    k = k.view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # (B, num_key_value_heads, T, head_dim)
    v = v.view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # (B, num_key_value_heads, T, head_dim)
    
    # Apply rotary positional embeddings to q and k (RoPE)
    q = self.rotary_emb.rotate_queries_or_keys(q)
    k = self.rotary_emb.rotate_queries_or_keys(k)
    
    # Repeat k and v for GQA (if num_key_value_heads < n_head)
    if self.num_key_value_heads != self.n_head:
        # Repeat k and v to match number of query heads
        repeat_factor = self.n_head // self.num_key_value_heads
        k = k.repeat_interleave(repeat_factor, dim=1)  # (B, n_head, T, head_dim)
        v = v.repeat_interleave(repeat_factor, dim=1)  # (B, n_head, T, head_dim)
    
    # Causal self-attention: (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
    att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v  # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)

    # Re-assemble all head outputs: (B, n_head, T, head_dim) -> (B, T, C)
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    
    # Output projection
    y = self.o_proj(y)
    return y
```

---

### Change 2: Update LlamaModel forward method (Cell 8)

**Step 2.1:** Find Cell 8 which contains `class LlamaModel`

**Step 2.2:** Find the `forward` method. Look for this code:
```python
def forward(self, idx, targets=None):
    # idx is of shape (B, T)
    B, T = idx.size()
    assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
    # forward the token and posisition embeddings
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
    pos_emb = self.pos_embed(pos) # position embeddings of shape (T, n_embd)
    tok_emb = self.embed_tokens(idx) # token embeddings of shape (B, T, n_embd)
    x = tok_emb + pos_emb
    # forward the blocks of the transformer
    x = self.transformer(x)
    # forward the final layernorm and the classifier
    x = self.ln_f(x)
    logits = self.lm_head(x) # (B, T, vocab_size)
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
```

**Replace with:**
```python
def forward(self, idx, targets=None):
    # idx is of shape (B, T)
    B, T = idx.size()
    assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
    # forward the token embeddings (RoPE is applied in attention, not here)
    x = self.embed_tokens(idx)  # token embeddings of shape (B, T, n_embd)
    # forward the blocks of the transformer
    x = self.transformer(x)
    # forward the final layernorm and the classifier
    x = self.ln_f(x)
    logits = self.lm_head(x)  # (B, T, vocab_size)
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
```

**Key change:** Remove the lines that create and add positional embeddings:
- Remove: `pos = torch.arange(0, T, dtype=torch.long, device=idx.device)`
- Remove: `pos_emb = self.pos_embed(pos)`
- Change: `x = tok_emb + pos_emb` to just `x = self.embed_tokens(idx)`

---

### Change 3: Ensure config has num_key_value_heads (Cell 4 or Cell 10)

**Step 3.1:** Find where you create the model config (likely in Cell 10 where you do `model_config = cfg.model`)

**Step 3.2:** Make sure you add `num_key_value_heads` to the config object. Look for this section:
```python
model_config = cfg.model
model_config.n_embd = model_config.hidden_size
model_config.n_head = model_config.num_attention_heads
model_config.block_size = model_config.max_position_embeddings
model_config.device = device
```

**Add this line:**
```python
model_config.num_key_value_heads = model_config.num_key_value_heads
```

Or if it's already accessible via `cfg.model.num_key_value_heads`, you can skip this step.

---

## Verification Steps

After making the changes:

1. **Re-run Cell 5** (LlamaAttention definition)
2. **Re-run Cell 8** (LlamaModel definition)
3. **Re-run Cell 10** (where you create the model)

4. **Check the model structure** - Run:
```python
print(model)
```

You should see:
- `(q_proj): Linear(in_features=576, out_features=576, bias=False)`
- `(k_proj): Linear(in_features=576, out_features=192, bias=False)`
- `(v_proj): Linear(in_features=576, out_features=192, bias=False)`
- `(o_proj): Linear(in_features=576, out_features=576, bias=False)`

**NOT:**
- `(c_attn): Linear(...)` ❌
- `(c_proj): Linear(...)` ❌

5. **Check parameter count** - Run your parameter counting function. It should show approximately **134,515,008** parameters (matching the reference model), not 176,166,720.

6. **Test forward pass** - Run a forward pass to ensure it works:
```python
logits, loss = model(x, y)
print(f"Loss: {loss.item():.4f}")
```

---

## Summary of Key Differences

| Aspect | Old (GPT-style) | New (LLaMA-style) |
|--------|----------------|-------------------|
| Attention | `c_attn` + `c_proj` | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Bias | `bias=True` | `bias=False` |
| Key/Value heads | 18 (same as query) | 6 (GQA - Grouped Query Attention) |
| Positional embeddings | Added to embeddings | Applied in attention (RoPE) |
| Parameters | ~176M | ~134M |

---

## Troubleshooting

**If you get an error about `num_key_value_heads`:**
- Make sure `config.yaml` has `num_key_value_heads: 6` under the `model` section
- Make sure it's accessible as `config.num_key_value_heads` in the attention class

**If you get shape mismatch errors:**
- Make sure you re-ran all the cells in order after making changes
- Check that `head_dim = config.n_embd // config.n_head` (should be 576 // 18 = 32)

**If parameter count is still wrong:**
- Check that all attention layers have `bias=False`
- Verify k_proj and v_proj output dimensions are 192 (6 heads × 32 head_dim)

