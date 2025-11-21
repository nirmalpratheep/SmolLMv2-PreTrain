# Fixes Needed to Match Reference Model

## 1. LlamaAttention Class (Cell 5)

Replace the entire class with:

```python
class LlamaAttention(nn.Module):
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

## 2. LlamaModel forward method (Cell 8)

Replace the forward method to remove positional embedding addition:

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

## Key Changes:
1. **Attention**: Separate q/k/v/o projections instead of combined c_attn
2. **GQA**: k_proj and v_proj use num_key_value_heads (192 features) instead of n_head
3. **Bias**: All attention layers have bias=False
4. **RoPE**: Applied in attention, not added to embeddings
5. **Positional embeddings**: Removed from forward pass (RoPE handles it)

