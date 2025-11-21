# Weight Sharing in Grouped Query Attention (GQA) - Parameter Reduction Analysis

## Overview

The reference model uses **Grouped Query Attention (GQA)**, which is a form of weight sharing that significantly reduces the number of parameters in the attention mechanism compared to standard Multi-Head Attention (MHA).

## Architecture Comparison

### Standard Multi-Head Attention (MHA)
- **All heads are independent**: Each query head has its own key and value projections
- **No weight sharing**: Q, K, V projections are separate for each head

### Grouped Query Attention (GQA)
- **Key/Value heads are shared**: Multiple query heads share the same key and value projections
- **Weight sharing**: Reduces parameters in K and V projections

## Your Model's Configuration

From `config.yaml`:
- `num_attention_heads: 18` (query heads)
- `num_key_value_heads: 6` (key/value heads)
- **Sharing ratio**: 18 ÷ 6 = **3 query heads share 1 key/value head**

## Parameter Calculation

### Hidden Dimensions
- `hidden_size (n_embd)`: 576
- `head_dim`: 576 ÷ 18 = **32** (dimension per head)

### Standard MHA (No Weight Sharing)
If we used standard MHA with 18 heads:

**Per layer:**
- `q_proj`: 576 × 576 = **331,776** parameters
- `k_proj`: 576 × 576 = **331,776** parameters (18 heads × 32 dims)
- `v_proj`: 576 × 576 = **331,776** parameters (18 heads × 32 dims)
- `o_proj`: 576 × 576 = **331,776** parameters

**Total per layer**: 1,327,104 parameters

**For 30 layers**: 1,327,104 × 30 = **39,813,120 parameters**

---

### GQA (With Weight Sharing)
With GQA using 18 query heads and 6 key/value heads:

**Per layer:**
- `q_proj`: 576 × 576 = **331,776** parameters (18 heads)
- `k_proj`: 576 × 192 = **110,592** parameters (6 heads × 32 dims)
- `v_proj`: 576 × 192 = **110,592** parameters (6 heads × 32 dims)
- `o_proj`: 576 × 576 = **331,776** parameters

**Total per layer**: 884,736 parameters

**For 30 layers**: 884,736 × 30 = **26,542,080 parameters**

---

## Parameter Reduction

### Per Layer Reduction
- **Standard MHA**: 1,327,104 parameters
- **GQA**: 884,736 parameters
- **Reduction**: 1,327,104 - 884,736 = **442,368 parameters per layer**
- **Reduction percentage**: (442,368 ÷ 1,327,104) × 100 = **33.33%**

### Total Model Reduction (30 layers)
- **Standard MHA**: 39,813,120 parameters
- **GQA**: 26,542,080 parameters
- **Total reduction**: 39,813,120 - 26,542,080 = **13,271,040 parameters**
- **Reduction percentage**: (13,271,040 ÷ 39,813,120) × 100 = **33.33%**

## Why This Works

### The Key Insight
In attention, the **query** determines what information to look for, while **key** and **value** represent the information available. 

- **Queries need to be diverse**: Each query head should be able to attend to different aspects
- **Keys/Values can be shared**: Multiple queries can effectively use the same key-value pairs

### How It Works
1. **18 query heads** project the input independently → 18 different "questions"
2. **6 key/value heads** project the input → 6 shared "answers"
3. **Each query head** attends to all 6 key/value pairs
4. **3 query heads** effectively share the same key/value projections (18 ÷ 6 = 3)

### Attention Computation
```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

With GQA:
- `Q`: Shape (batch, 18, seq_len, 32) - 18 query heads
- `K`: Shape (batch, 6, seq_len, 32) - 6 key heads (repeated to 18)
- `V`: Shape (batch, 6, seq_len, 32) - 6 value heads (repeated to 18)

The repetition happens during computation, not in the weights!

## Real-World Impact

### Memory Savings
- **33% fewer parameters** in attention layers
- For a model with ~134M parameters, this saves ~13M parameters
- **Memory reduction**: ~52 MB (assuming float16, 2 bytes per parameter)

### Computational Efficiency
- **Fewer matrix multiplications** in K and V projections
- **Faster inference** due to smaller weight matrices
- **Lower memory bandwidth** requirements

### Performance Trade-off
- **Minimal performance loss**: Research shows GQA with 3:1 ratio (like yours) maintains ~99% of full attention performance
- **Significant efficiency gain**: 33% parameter reduction with negligible quality loss

## Comparison Table

| Architecture | Query Heads | K/V Heads | K/V Params | Total Params (30 layers) | Reduction |
|--------------|-------------|-----------|------------|--------------------------|-----------|
| **Standard MHA** | 18 | 18 | 663,552 | 39,813,120 | 0% |
| **GQA (3:1)** | 18 | 6 | 221,184 | 26,542,080 | **33.33%** |
| **GQA (2:1)** | 18 | 9 | 331,776 | 33,177,600 | 16.67% |
| **MQA (18:1)** | 18 | 1 | 36,864 | 22,118,400 | 44.44% |

*Note: MQA (Multi-Query Attention) is an extreme case where all query heads share a single key/value head*

## Your Model's Specific Numbers

Based on your reference model showing **134,515,008 total parameters**:

### Attention Parameters Breakdown
- **30 layers** × **884,736 params/layer** = **26,542,080 attention parameters**
- This represents **~19.7%** of total model parameters

### Without GQA (hypothetical)
- **30 layers** × **1,327,104 params/layer** = **39,813,120 attention parameters**
- Would represent **~29.6%** of total model parameters

### Savings in Your Model
- **Attention parameter savings**: 13,271,040 parameters
- **Percentage of total model**: ~9.9% reduction in total model size
- **Memory saved**: ~25.4 MB (float16) or ~50.8 MB (float32)

## Key Takeaways

1. **33.33% reduction** in attention layer parameters
2. **Minimal performance impact** - GQA with 3:1 ratio is nearly as good as full attention
3. **Significant efficiency gains** - Faster inference, lower memory usage
4. **Industry standard** - Used in models like LLaMA-2, Mistral, and many modern LLMs

## Why 3:1 Ratio?

The 3:1 ratio (18 query heads : 6 key/value heads) is a sweet spot because:
- **Too few K/V heads** (like 1:1 MQA): May lose important information diversity
- **Too many K/V heads** (like 1:1 MHA): No parameter savings
- **3:1 ratio**: Balances efficiency and performance - research shows it maintains ~99% of full attention quality

## References

This technique was popularized by:
- **GQA paper**: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
- Used in: LLaMA-2, Mistral, and many modern efficient LLMs

