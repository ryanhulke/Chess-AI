from dataclasses import dataclass
import json
from typing import Optional, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(var + self.eps)) * self.weight

class QKNorm(nn.Module):
    def __init__(self, d_head: int, eps: float = 1e-6):
        super().__init__()
        self.q = RMSNorm(d_head, eps)
        self.k = RMSNorm(d_head, eps)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        return self.q(q), self.k(k)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mult: float = 2.6667, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        inner = int(round(mult * d_model))
        self.wg = nn.Linear(d_model, inner, bias=bias)
        self.wv = nn.Linear(d_model, inner, bias=bias)
        self.wo = nn.Linear(inner, d_model, bias=bias)
        self.drop = nn.Dropout(float(dropout)) if dropout and dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.wo(F.silu(self.wg(x)) * self.wv(x)))


@dataclass
class PredictorConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    output_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    rotary_n_embd: Optional[int] = None
    n_kv_head: Optional[int] = None
    gqa_groups: int = 1
    mlp_mult: float = 2.6667
    bias: bool = False
    dropout: float = 0.0
    tie_embeddings: bool = True
    return_dict: bool = False

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        head_dim = self.n_embd // self.n_head
        if self.rotary_n_embd is None:
            self.rotary_n_embd = head_dim
        if not (0 < int(self.rotary_n_embd) <= head_dim):
            raise ValueError("rotary_n_embd must be in (0, head_dim]")
        if self.n_kv_head is None:
            self.n_kv_head = max(1, self.n_head // max(1, self.gqa_groups))
        if self.n_head % int(self.n_kv_head) != 0:
            raise ValueError("n_head must be divisible by n_kv_head")

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


class BidirectionalSelfAttention(nn.Module):
    def __init__(self, config: PredictorConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.rotary_dim = int(config.rotary_n_embd)
        self.dropout_p = float(config.dropout)
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.rotary_pos_emb = RotaryEmbedding(self.rotary_dim)
        self.qk_norm = QKNorm(self.head_dim)
        self.out_drop = nn.Dropout(self.dropout_p) if self.dropout_p > 0.0 else nn.Identity()

    def apply_rope(self, q: torch.Tensor, k: torch.Tensor):
        if self.rotary_dim == self.head_dim:
            return self.rotary_pos_emb.rotate_queries_or_keys(q), self.rotary_pos_emb.rotate_queries_or_keys(k)
        q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
        k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
        q_rot = self.rotary_pos_emb.rotate_queries_or_keys(q_rot)
        k_rot = self.rotary_pos_emb.rotate_queries_or_keys(k_rot)
        return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        h = self.n_head
        hk = self.n_kv_head
        d = self.head_dim

        q = self.q_proj(x).view(b, t, h, d).transpose(1, 2)
        k = self.k_proj(x).view(b, t, hk, d).transpose(1, 2)
        v = self.v_proj(x).view(b, t, hk, d).transpose(1, 2)

        q, k = self.apply_rope(q, k)
        q, k = self.qk_norm(q, k)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
            enable_gqa=True,
        )

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.out_drop(self.o_proj(y))



class ParallelBlock(nn.Module):
    def __init__(self, cfg: PredictorConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.n_embd)
        self.attn = BidirectionalSelfAttention(cfg)
        self.mlp = SwiGLU(cfg.n_embd, mult=cfg.mlp_mult, bias=cfg.bias, dropout=cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        return x + self.attn(y) + self.mlp(y)


class BidirectionalPredictor(nn.Module):
    def __init__(self, cfg: PredictorConfig):
        super().__init__()
        self.config = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.emb_drop = nn.Dropout(float(cfg.dropout)) if cfg.dropout and cfg.dropout > 0.0 else nn.Identity()
        self.blocks = nn.ModuleList([ParallelBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd)

        self.policy_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.value_head = nn.Linear(cfg.n_embd, cfg.output_size, bias=False)

        if cfg.tie_embeddings:
            self.policy_head.weight = self.wte.weight

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        ids: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        b, t = ids.size()
        if t > int(self.config.block_size):
            raise ValueError(f"seq len {t} exceeds block_size {self.config.block_size}")

        x = self.emb_drop(self.wte(ids))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)

        policy_logits = self.policy_head(x)
        value_logits = self.value_head(x)

        use_dict = self.config.return_dict if return_dict is None else bool(return_dict)
        if use_dict:
            return {"policy_logits": policy_logits, "value_logits": value_logits}
        return value_logits