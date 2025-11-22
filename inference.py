"""
Simple Inference Script for SmolLM v2
Load a trained checkpoint and generate text from sample prompts.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import yaml
from transformers import AutoTokenizer
from rotary_embedding_torch import RotaryEmbedding

# ============================================================================
# Model Architecture (copied from notebook for self-contained script)
# ============================================================================

class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.n_embd // config.n_head
        
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        
        if self.num_key_value_heads != self.n_head:
            repeat_factor = self.n_head // self.num_key_value_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        return y

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        
        if config.hidden_act == "silu":
            self.act_fn = nn.SiLU()
        elif config.hidden_act == "gelu":
            self.act_fn = nn.GELU()
        elif config.hidden_act == "relu":
            self.act_fn = nn.ReLU()
    
    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class LlamaRMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.variance_epsilon = float(config.rms_norm_eps)
    
    def forward(self, x):
        return self.weight * x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.variance_epsilon)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LlamaRMSNorm(config)
        self.attn = LlamaAttention(config)
        self.ln_2 = LlamaRMSNorm(config)
        self.mlp = LlamaMLP(config) 

    def forward(self, x):
        residual = x
        x = self.ln_1(x)
        x = x + self.attn(x)
        x = self.ln_2(x)
        x = x + self.mlp(x)
        x = x + residual
        return x

class LlamaDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embed = RotaryEmbedding(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.ln_f = LlamaRMSNorm(config)
        self.transformer = LlamaDecoder(config)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        x = self.embed_tokens(idx)
        x = self.transformer(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, input_ids, max_new_tokens):
        """Generate new tokens up to max_new_tokens."""
        self.eval()
        input_ids = input_ids.to(self.config.device)
        generated_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            x = self.embed_tokens(generated_ids)
            x = self.transformer(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            logits = logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        return generated_ids

# ============================================================================
# Inference Functions
# ============================================================================

# Sample prompts
SAMPLE_PROMPTS = [
    "The weather is",
    "Once upon a time",
    "In a world where",
    "The future of artificial intelligence",
    "To be or not to be",
    "The quick brown fox",
    "Hello, how are you",
    "The meaning of life is",
]

def load_config_from_yaml():
    """Load model configuration from config.yaml."""
    with open("config.yaml", 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict

def create_model_config_from_yaml(config_dict, device='cpu'):
    """Create model config object from config.yaml."""
    class SimpleConfig:
        pass
    
    model_config = SimpleConfig()
    model_yaml = config_dict['model']
    
    # Set all config values from config.yaml
    model_config.hidden_size = model_yaml['hidden_size']
    model_config.n_embd = model_yaml['hidden_size']  # n_embd = hidden_size
    model_config.n_head = model_yaml['num_attention_heads']
    model_config.block_size = model_yaml['max_position_embeddings']
    model_config.device = device
    model_config.num_key_value_heads = model_yaml['num_key_value_heads']
    model_config.vocab_size = model_yaml['vocab_size']
    model_config.num_hidden_layers = model_yaml['num_hidden_layers']
    model_config.intermediate_size = model_yaml['intermediate_size']
    model_config.rms_norm_eps = model_yaml['rms_norm_eps']
    model_config.hidden_act = model_yaml['hidden_act']
    model_config.mlp_bias = model_yaml['mlp_bias']
    
    return model_config

def load_model_from_checkpoint(checkpoint_path, config_dict, device='cpu'):
    """Load model from checkpoint file using config.yaml for architecture."""
    print(f"Loading configuration from config.yaml...")
    
    # Create model config from config.yaml
    model_config = create_model_config_from_yaml(config_dict, device)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with architecture from config.yaml
    model = LlamaModel(model_config)
    
    # Load weights from checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully! (Trained for {checkpoint['step']} steps)")
    return model

def load_tokenizer():
    """Load tokenizer from config."""
    with open("config.yaml", 'r') as f:
        config_dict = yaml.safe_load(f)
    tokenizer = AutoTokenizer.from_pretrained(
        config_dict["tokenizer"]["tokenizer_name_or_path"],
        use_fast=config_dict["tokenizer"]["use_fast"]
    )
    return tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=100, device='cpu'):
    """Generate text from a prompt."""
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(prompt_ids, max_new_tokens=max_new_tokens)
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

def main():
    """Main inference function."""
    print("=" * 80)
    print("SmolLM v2 Inference")
    print("=" * 80)
    
    # Load configuration from config.yaml
    try:
        config_dict = load_config_from_yaml()
        print("Configuration loaded from config.yaml")
    except FileNotFoundError:
        print("Error: config.yaml not found!")
        print("Please make sure config.yaml exists in the current directory.")
        return
    except Exception as e:
        print(f"Error loading config.yaml: {e}")
        return
    
    # Device selection
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU")
    else:
        print("Using CPU")
    
    # Load checkpoint (using config.yaml for architecture)
    checkpoint_path = "checkpoint_step_5050.pt"
    try:
        model = load_model_from_checkpoint(checkpoint_path, config_dict, device)
    except FileNotFoundError:
        checkpoint_path = "checkpoint_step_5000.pt"
        try:
            model = load_model_from_checkpoint(checkpoint_path, config_dict, device)
        except FileNotFoundError:
            print(f"\nError: Could not find checkpoint file.")
            print("Please make sure you have trained the model and saved a checkpoint.")
            return
    
    # Load tokenizer from config.yaml
    tokenizer = load_tokenizer()
    
    print("\n" + "=" * 80)
    print("Sample Prompts:")
    print("=" * 80)
    for i, prompt in enumerate(SAMPLE_PROMPTS, 1):
        print(f"{i}. {prompt}")
    print("=" * 80)
    
    # Interactive loop
    while True:
        print("\nOptions:")
        print("1. Select a sample prompt (1-8)")
        print("2. Enter your own prompt")
        print("3. Exit")
        
        choice = input("\nYour choice: ").strip()
        
        if choice == '3':
            print("Goodbye!")
            break
        elif choice in ['1', '2']:
            if choice == '1':
                prompt_num = input("Enter prompt number (1-8): ").strip()
                try:
                    prompt_idx = int(prompt_num) - 1
                    if 0 <= prompt_idx < len(SAMPLE_PROMPTS):
                        prompt = SAMPLE_PROMPTS[prompt_idx]
                    else:
                        print("Invalid prompt number!")
                        continue
                except ValueError:
                    print("Invalid input!")
                    continue
            else:
                prompt = input("Enter your prompt: ").strip()
                if not prompt:
                    print("Prompt cannot be empty!")
                    continue
            
            # Get max tokens
            max_tokens_input = input("Max tokens to generate (default 100): ").strip()
            max_tokens = int(max_tokens_input) if max_tokens_input else 100
            
            # Generate
            print(f"\nGenerating text...")
            print(f"Prompt: '{prompt}'")
            print("-" * 80)
            
            generated = generate_text(model, tokenizer, prompt, max_new_tokens=max_tokens, device=device)
            
            print(f"Generated: '{generated}'")
            print("=" * 80)
        else:
            print("Invalid choice! Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
