# SmolLM v2 Pretraining - From Scratch Implementation

This repository contains a complete implementation of **SmolLM v2 (135M parameters)** from scratch, including the model architecture, training loop, and all necessary components. This is a faithful recreation of the HuggingFace SmolLM2-135M model architecture.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Recreating SmolLM v2 from Scratch](#recreating-smollm-v2-from-scratch)
- [Model Components](#model-components)
- [Training from Scratch](#training-from-scratch)
- [Usage](#usage)
- [Model Details](#model-details)

## Overview

SmolLM v2 is a small language model with 135M parameters, based on the LLaMA architecture. This implementation recreates the model from scratch using PyTorch, including:

- **Grouped Query Attention (GQA)** for efficient attention computation
- **Rotary Position Embeddings (RoPE)** for positional encoding
- **RMSNorm** for layer normalization
- **SiLU activation** in MLP layers
- **Weight tying** between embedding and output layers

### Model Specifications

- **Parameters**: 134,515,008 (135M)
- **Hidden Size**: 576
- **Layers**: 30
- **Attention Heads**: 18
- **Key-Value Heads**: 6 (GQA ratio: 3:1)
- **Vocabulary Size**: 49,152
- **Max Sequence Length**: 8,192
- **Intermediate Size**: 1,536

## Quick Start

To quickly get started with training SmolLM v2 from scratch:

1. **Install dependencies:**
   ```bash
   pip install rotary-embedding-torch transformers pyyaml torch
   ```

2. **Open the Jupyter notebook:**
   ```bash
   jupyter notebook smolLMV2-pretrain.ipynb
   ```

3. **Run cells sequentially:**
   - Cell 0: Install and import dependencies
   - Cell 2: Load tokenizer
   - Cell 5: Load configuration from `config.yaml`
   - Cell 6-9: Define model architecture components
   - Cell 11: Create and test model
   - Cell 15: Start training

4. **Prepare your data:**
   - Place training text in `input.txt`
   - The notebook will automatically tokenize it

5. **Train:**
   - Adjust hyperparameters in Cell 15 (batch size, sequence length, learning rate)
   - Run the training loop

For detailed step-by-step instructions, see [Recreating SmolLM v2 from Scratch](#recreating-smollm-v2-from-scratch).

## Prerequisites

- Python 3.13 or higher
- PyTorch 2.0 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended)
- 2GB+ disk space for model weights

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd smolLMV2-pretrain
```

### 2. Install Dependencies

```bash
pip install rotary-embedding-torch transformers pyyaml torch
```

Or install from the project:

```bash
pip install -e .
```

### Required Packages

- `torch` >= 2.0.0
- `transformers` >= 4.21.0
- `rotary-embedding-torch` >= 0.8.0
- `pyyaml` >= 6.0
- `einops` >= 0.7.0 (dependency of rotary-embedding-torch)

## Project Structure

```
smolLMV2-pretrain/
├── README.md                 # This file
├── config.yaml               # Model and training configuration
├── main.py                   # Main entry point (placeholder)
├── pyproject.toml            # Project metadata
├── input.txt                 # Training data (text corpus)
├── smolLMV2-pretrain.ipynb   # Jupyter notebook with full implementation
└── log/
    └── profiler/             # Training profiler logs
```

## Configuration

The model configuration is defined in `config.yaml`. Key parameters:

### Model Configuration

```yaml
model:
  model_type: "llama"
  vocab_size: 49152
  hidden_size: 576
  intermediate_size: 1536
  num_hidden_layers: 30
  num_attention_heads: 18
  num_key_value_heads: 6  # GQA: 18/6 = 3:1 ratio
  max_position_embeddings: 8192
  rms_norm_eps: 1e-05
  hidden_act: "silu"
  mlp_bias: false
```

### Tokenizer Configuration

```yaml
tokenizer:
  tokenizer_name_or_path: "HuggingFaceTB/SmolLM2-135M"
  use_fast: true
  padding_side: "right"
  truncation_side: "right"
```

## Model Components

### 1. LlamaAttention (Grouped Query Attention)

Implements GQA with the following structure:
- **Query projection**: All 18 heads
- **Key/Value projections**: Only 6 heads (shared across 3 query heads each)
- **RoPE**: Applied to queries and keys
- **Causal masking**: Prevents attention to future tokens

```python
class LlamaAttention(nn.Module):
    - q_proj: Linear(576, 576)      # 18 heads × 32 dims
    - k_proj: Linear(576, 192)      # 6 heads × 32 dims
    - v_proj: Linear(576, 192)      # 6 heads × 32 dims
    - o_proj: Linear(576, 576)
    - rotary_emb: RotaryEmbedding(32)
```

### 2. LlamaMLP (Feed-Forward Network)

Uses SwiGLU activation (SiLU-gated linear unit):
- **Gate projection**: `SiLU(gate_proj(x))`
- **Up projection**: `up_proj(x)`
- **Output**: `down_proj(gate * up)`

```python
class LlamaMLP(nn.Module):
    - gate_proj: Linear(576, 1536, bias=False)
    - up_proj: Linear(576, 1536, bias=False)
    - down_proj: Linear(1536, 576, bias=False)
    - act_fn: SiLU()
```

### 3. LlamaRMSNorm

Root Mean Square Layer Normalization:
- Normalizes across the hidden dimension
- Uses learnable scale parameter
- Epsilon: 1e-05

### 4. LlamaDecoderLayer

Standard transformer decoder layer:
```
x = x + attention(rms_norm_1(x))
x = x + mlp(rms_norm_2(x))
```

### 5. LlamaModel

Complete model architecture:
- **Token Embedding**: `Embedding(49152, 576)`
- **30 Decoder Layers**: Each with attention and MLP
- **Final Layer Norm**: RMSNorm
- **Language Model Head**: `Linear(576, 49152)` (weight-tied with embeddings)

## Recreating SmolLM v2 from Scratch

This section provides a complete step-by-step guide to recreate SmolLM v2 from scratch, starting with an empty repository.

### Complete Setup Process

#### Step 1: Environment Setup

1. **Create a new directory:**
   ```bash
   mkdir smolLMV2-pretrain
   cd smolLMV2-pretrain
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch transformers rotary-embedding-torch pyyaml einops
   ```

#### Step 2: Create Configuration File

Create `config.yaml` with the model architecture specifications:

```yaml
# SmolLMV2 Pretraining Configuration
model:
  model_type: "llama"
  vocab_size: 49152
  hidden_size: 576
  intermediate_size: 1536
  num_hidden_layers: 30
  num_attention_heads: 18
  num_key_value_heads: 6
  max_position_embeddings: 8192
  rms_norm_eps: 1e-05
  use_cache: true
  pad_token_id: null
  bos_token_id: 1
  eos_token_id: 2
  mlp_bias: false
  mlp_gate_proj_features: 1536
  mlp_up_proj_features: 1536
  mlp_down_proj_features: 576
  hidden_act: "silu"
  use_silu: true
  pretrained_model_name_or_path: "HuggingFaceTB/SmolLM2-135M"
  torch_dtype: "float16"
  device_map: "auto"
  trust_remote_code: false

tokenizer:
  tokenizer_name_or_path: "HuggingFaceTB/SmolLM2-135M"
  use_fast: true
  padding_side: "right"
  truncation_side: "right"
```

#### Step 3: Implement Model Architecture

The model consists of several components. Here's the order to implement them:

1. **LlamaRMSNorm** - Root Mean Square Layer Normalization
2. **LlamaAttention** - Grouped Query Attention with RoPE
3. **LlamaMLP** - SwiGLU feed-forward network
4. **LlamaDecoderLayer** - Transformer decoder layer
5. **LlamaDecoder** - Stack of decoder layers
6. **LlamaModel** - Complete model with embeddings and language model head

All components are implemented in `smolLMV2-pretrain.ipynb`. Key architectural decisions:

- **GQA**: 18 query heads share 6 key-value heads (3:1 ratio)
- **RoPE**: Applied in attention layer, not as separate positional embeddings
- **Weight Tying**: Embedding and output layers share weights
- **Initialization**: Normal distribution (mean=0, std=0.02) for all weights

#### Step 4: Prepare Training Data

Place your training corpus in `input.txt`. The text will be tokenized using the SmolLM2 tokenizer.

## Training from Scratch

### Step 1: Prepare Training Data

Place your training text corpus in `input.txt`. The data will be tokenized using the SmolLM2 tokenizer.

### Step 2: Initialize Model

The model can be initialized from scratch with random weights. The model classes are defined in the Jupyter notebook. You can either:

**Option A: Use the Jupyter Notebook**
Run the cells in `smolLMV2-pretrain.ipynb` sequentially to build and train the model.

**Option B: Extract Code to Python Script**
Copy the model classes from the notebook into a Python module, then:

```python
from transformers import AutoTokenizer
import yaml
import torch
# Import your model classes (LlamaModel, ModelConfig, etc.)

# Load configuration
with open("config.yaml", 'r') as f:
    config_dict = yaml.safe_load(f)

# Create model config (add compatibility attributes)
model_config = type('Config', (), config_dict["model"])()
model_config.n_embd = model_config.hidden_size
model_config.n_head = model_config.num_attention_heads
model_config.block_size = model_config.max_position_embeddings
model_config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_config.num_key_value_heads = model_config.num_key_value_heads

# Initialize model
model = LlamaModel(model_config)
model.to(model_config.device)
```

### Step 3: Setup Data Loader

```python
class DataLoaderLite:
    def __init__(self, B, T, tokenizer):
        self.B = B  # batch size
        self.T = T  # sequence length
        self.tokenizer = tokenizer
        
        # Load and tokenize text
        with open('input.txt', 'r') as f:
            text = f.read()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        self.tokens = torch.tensor(tokens)
        self.current_position = 0
    
    def next_batch(self):
        buf = self.tokens[self.current_position: self.current_position + self.B * self.T + 1]
        x = buf[:-1].view(self.B, self.T)  # inputs
        y = buf[1:].view(self.B, self.T)   # targets (shifted by 1)
        self.current_position += self.B * self.T
        if self.current_position + (self.B * self.T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
```

### Step 4: Training Loop

```python
# Set seed for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Training hyperparameters
B = 8              # batch size
T = 1024           # sequence length
learning_rate = 3e-4
num_steps = 10000  # adjust based on your dataset size

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
train_loader = DataLoaderLite(B=B, T=T, tokenizer=tokenizer)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for step in range(num_steps):
    # Get batch
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    
    # Forward pass
    optimizer.zero_grad()
    logits, loss = model(x, y)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Logging
    if step % 100 == 0:
        print(f'Step {step} | Loss: {loss.item():.4f}')
```

### Step 5: Save Model

```python
# Save model state
torch.save({
    'model_state_dict': model.state_dict(),
    'config': model_config.__dict__,
    'optimizer_state_dict': optimizer.state_dict(),
}, 'smollm_v2_135m.pt')
```

## Usage

### Loading the Model

```python
import torch
# Import your model classes (LlamaModel, etc.) from the notebook or extracted module

# Load saved model
checkpoint = torch.load('smollm_v2_135m.pt')
model = LlamaModel(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Text Generation

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

# Encode input
text = "The weather is"
input_ids = tokenizer.encode(text, return_tensors='pt')

# Generate
model.eval()
with torch.no_grad():
    generated_ids = model.generate(input_ids, max_new_tokens=50)

# Decode
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)
```

### Using the Jupyter Notebook

The `smolLMV2-pretrain.ipynb` notebook contains:
1. Model architecture implementation
2. Configuration loading from YAML
3. Training loop with profiling
4. Model inspection and parameter counting
5. Step-by-step explanations

Run the notebook cells sequentially to:
- Install dependencies
- Load configuration
- Build the model
- Train the model
- Generate text

## Model Details

### Parameter Breakdown

| Component | Parameters |
|-----------|------------|
| Token Embeddings | 28,311,552 |
| Attention (per layer) | 884,736 |
| MLP (per layer) | 2,654,208 |
| Layer Norms (per layer) | 1,152 |
| Final Layer Norm | 576 |
| **Total (30 layers)** | **134,515,008** |

### Architecture Highlights

1. **Grouped Query Attention (GQA)**
   - Reduces memory usage by sharing key-value heads
   - 18 query heads share 6 key-value heads (3:1 ratio)
   - Maintains model quality while improving efficiency

2. **Rotary Position Embeddings (RoPE)**
   - Applied directly to queries and keys in attention
   - Provides relative positional information
   - Enables longer context lengths

3. **RMSNorm**
   - Simpler than LayerNorm (no mean centering)
   - Faster computation
   - Standard in modern LLMs

4. **Weight Tying**
   - Embedding and output layers share weights
   - Reduces parameters and improves training stability

5. **SwiGLU Activation**
   - SiLU-gated linear unit
   - Better than standard ReLU/GELU in practice
   - Formula: `SiLU(gate(x)) * up(x)`

### Initialization

Weights are initialized using:
- **Linear layers**: Normal distribution (mean=0.0, std=0.02)
- **Embeddings**: Normal distribution (mean=0.0, std=0.02)
- **Biases**: Zero initialization (if used)

### Training Recommendations

- **Batch Size**: 8-32 (adjust based on GPU memory)
- **Sequence Length**: 1024-2048 (up to 8192 max)
- **Learning Rate**: 3e-4 (standard for AdamW)
- **Optimizer**: AdamW with default betas (0.9, 0.999)
- **Gradient Clipping**: Consider adding for stability (max_norm=1.0)
- **Mixed Precision**: Use float16 or bfloat16 for faster training

### Expected Training Metrics

- **Initial Loss**: ~10.8 (random prediction baseline)
- **Target Loss**: < 2.0 (after sufficient training)
- **Training Speed**: 
  - CPU: ~100-500 tokens/sec
  - GPU (V100): ~5,000-10,000 tokens/sec
  - GPU (A100): ~20,000-50,000 tokens/sec

## Notes

- The model architecture matches HuggingFace's SmolLM2-135M implementation
- Tokenizer is loaded from HuggingFace (not recreated)
- Training data format: plain text file, one document per line recommended
- Random seed: 1337 (for reproducibility)
- The complete implementation is in `smolLMV2-pretrain.ipynb` - run cells sequentially
- For production use, consider adding:
  - Gradient accumulation
  - Learning rate scheduling (cosine decay, warmup)
  - Checkpointing (save every N steps)
  - Distributed training support (DDP, FSDP)
  - Evaluation metrics (perplexity, validation loss)
  - Mixed precision training (AMP)
  - Gradient clipping

## References

- [SmolLM v2 Model Card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- [LLaMA Paper](https://arxiv.org/abs/2302.13971)
- [RoPE Paper](https://arxiv.org/abs/2104.09864)
- [GQA Paper](https://arxiv.org/abs/2305.13245)

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

