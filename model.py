import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from torch import Tensor, nn
from transformers import AutoTokenizer


@dataclass
class Qwen3Config:
    attention_dropout: float = 0.0
    attention_bias: bool = False
    eos_token_id: int = 151_643
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 4096
    intermediate_size: int = 12_288
    max_position_embeddings: int = 32_768
    num_attention_heads: int = 32
    num_hidden_layers: int = 36
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-06
    rope_theta: int = 1_000_000
    torch_dtype: str = "bfloat16"
    vocab_size: int = 151_936


def load_config(config_path: str) -> Qwen3Config:
    with open(config_path, "r") as f:
        config = json.load(f)
    valid_fields = {
        k: v for k, v in config.items() if k in Qwen3Config.__dataclass_fields__
    }
    return Qwen3Config(**valid_fields)


class MLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = getattr(F, config.hidden_act)

    def forward(self, input: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(input)) * self.up_proj(input))


class RMSNorm(nn.Module):
    def __init__(self, config: Qwen3Config, shape):
        super().__init__()
        self.eps = config.rms_norm_eps
        self.weight = nn.Parameter(torch.ones(shape))

    def forward(self, input: Tensor) -> Tensor:
        input_type = input.dtype
        input = input.to(torch.float32)
        return F.rms_norm(input, self.weight.shape, self.weight, self.eps).to(
            input_type
        )


class RotaryEmbedding(nn.Module):
    """
    Implements rotary position embedding.
    Pairwise elements form a complex number z = a+bi which is rotated by angle mθ
    m = position index, θ = rotation frequency
    R(z,mθ) = [a*cos(mθ) - b*sin(mθ), a*sin(mθ) + b*cos(mθ)] = z*cos(mθ) + rotate_half(z)*sin(mθ)
    where rotate_90(z) = -b+ai performs a 90 degree counter-clockwise rotation
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.base = config.rope_theta
        self.dim = config.head_dim

        pair_indices = torch.arange(0, self.dim, 2).float()
        theta = 1.0 / (self.base ** (pair_indices / self.dim))

        position_indices = torch.arange(config.max_position_embeddings)

        rot_map = torch.outer(position_indices, theta)

        angles = torch.cat((rot_map, rot_map), dim=-1)

        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def _rotate_90(self, input: Tensor) -> Tensor:
        first_half = input[..., : self.dim // 2]
        second_half = input[..., self.dim // 2 :]
        return torch.cat((-second_half, first_half), dim=-1)

    @torch.no_grad()
    def forward(self, input: Tensor) -> Tensor:
        sequence_length = input.shape[2]

        cos_for_seq = self.cos_cached[:sequence_length]
        sin_for_seq = self.sin_cached[:sequence_length]

        rotated_input = (input * cos_for_seq.to(input.dtype)) + (
            self._rotate_90(input) * sin_for_seq.to(input.dtype)
        )
        return rotated_input


class SelfAttention(nn.Module):
    """Grouped Query Attention with Rotary Embeddings and QK Norm."""

    def __init__(self, config: Qwen3Config, rope: RotaryEmbedding):
        super().__init__()
        self.config = config
        self.rope = rope
        self.q_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        # QK norm on head dimension not hidden size
        self.q_norm = RMSNorm(config, config.head_dim)
        self.k_norm = RMSNorm(config, config.head_dim)

    def forward(self, input: Tensor) -> Tensor:
        batch_size, seq_len, _ = input.shape
        q = self.q_proj(input).view(
            batch_size, seq_len, self.config.num_attention_heads, self.config.head_dim
        )
        k = self.k_proj(input).view(
            batch_size, seq_len, self.config.num_key_value_heads, self.config.head_dim
        )
        v = self.v_proj(input).view(
            batch_size, seq_len, self.config.num_key_value_heads, self.config.head_dim
        )

        # (batch, num_heads, seq_len, head_dim)
        q_normed = self.q_norm(q).transpose(1, 2)
        k_normed = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q_rope = self.rope(q_normed)
        k_rope = self.rope(k_normed)

        attn_out = nn.functional.scaled_dot_product_attention(
            q_rope,
            k_rope,
            v,
            is_causal=True,
            dropout_p=self.config.attention_dropout,
            enable_gqa=True,
        )  # (batch, num_q_heads, seq_len, head_dim)

        attn_out = attn_out.transpose(1, 2).reshape(
            batch_size, seq_len, -1
        )  # (batch, seq_len, hidden_size)

        return self.o_proj(attn_out)


class DecoderBlock(nn.Module):
    def __init__(self, config: Qwen3Config, rope: RotaryEmbedding):
        super().__init__()
        self.config = config
        self.self_attn = SelfAttention(config, rope)
        self.mlp = MLP(config)

        self.input_layernorm = RMSNorm(config, config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config, config.hidden_size)

    def forward(self, input: Tensor) -> Tensor:
        residual = input

        input_norm = self.input_layernorm(input)

        attn_out = self.self_attn(input_norm)

        residual_post_attn = attn_out + residual

        post_attn_norm = self.post_attention_layernorm(residual_post_attn)

        mlp_out = self.mlp(post_attn_norm)

        return mlp_out + residual_post_attn


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        rope = RotaryEmbedding(config)
        self.model = nn.ModuleDict(
            {
                "embed_tokens": nn.Embedding(config.vocab_size, config.hidden_size),
                "layers": nn.ModuleList(
                    [
                        DecoderBlock(config, rope)
                        for _ in range(config.num_hidden_layers)
                    ]
                ),
                "norm": RMSNorm(config, config.hidden_size),
            }
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input: Tensor) -> Tensor:
        hidden_states = self.model["embed_tokens"](input)

        for layer in self.model["layers"]:
            hidden_states = layer(hidden_states)

        norm = self.model["norm"](hidden_states)

        return self.lm_head(norm)

    @classmethod
    def from_pretrained(cls, config: Qwen3Config, tensors_dir: str) -> "Qwen3Model":
        with init_empty_weights():
            model = cls(config)
        model = load_checkpoint_and_dispatch(
            model,
            tensors_dir,
            device_map={"": 0 if torch.cuda.is_available() else "cpu"},
            dtype=config.torch_dtype,
        )
        return model

    @torch.no_grad()
    def generate(
        self,
        tokenizer,
        prompt: str,
        stream: bool = True,
        max_tokens: int | None = 100,
        sampling: bool = True,
        temperature: float = 0.3,
    ) -> str:
        """
        Args:
            tokenizer: The tokenizer to use for encoding the prompt.
            prompt: The prompt to generate text from.
            stream: Whether to print the text as it is generated.
            max_tokens: The maximum number of tokens.
            sampling: Whether to use sampling instead of greedy decoding.
            temperature: The temperature for sampling.
        Returns:
            The generated text including the initial prompt.

        Generates text from a prompt using greedy decoding or sampling.
        If stream is True, tokens are printed as they are generated.
        Generation stops when an EOS token is produced or max_position_embeddings is reached.
        """
        self.eval()
        device = next(self.parameters()).device
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        max_tokens = max_tokens or self.config.max_position_embeddings
        while input_ids.shape[-1] < max_tokens:
            logits = self(input_ids)
            next_token_logits = logits[:, -1, :]

            if sampling and temperature > 0:
                next_token_logits = next_token_logits / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if next_token.item() == self.config.eos_token_id:
                break

            if stream:
                print(tokenizer.decode(next_token[0]), end="", flush=True)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

        if stream:
            print()

        return tokenizer.decode(input_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    config = load_config("config.json")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Base")
    model = Qwen3Model.from_pretrained(config, "qwen3_weights/")
    print("**Greedy** \n")
    model.generate(
        tokenizer,
        "Top 3 Michelin Restaurants:",
        stream=True,
        sampling=False,
        max_tokens=None,
    )
    print("\n**Sampling** \n")
    model.generate(
        tokenizer,
        "What is the capital of USA?",
        stream=True,
        sampling=True,
        temperature=0.6,
        max_tokens=100,
    )
