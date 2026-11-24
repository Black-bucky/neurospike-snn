import torch
import torch.nn as nn
from typing import Optional, Tuple
from .spiking_cnn import LIFNode


class SpikeAttention(nn.Module):
    r"""
    Spike-driven Multi-Head Self-Attention (Spike-DMSA).
    Unlike conventional softmax attention, spike-driven attention uses
    integer/binary spike matrix multiplications:
        Attn = (Q_{spk} \times K_{spk}^T) \times V_{spk}
    eliminating costly FP32 softmax and enabling neuromorphic integer addition.
    """
    def __init__(self, dim: int, num_heads: int = 4, beta: float = 0.9):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.q_lif = LIFNode(beta=beta)
        self.k_lif = LIFNode(beta=beta)
        self.v_lif = LIFNode(beta=beta)

        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.out_lif = LIFNode(beta=beta)

    def forward(
        self,
        x: torch.Tensor,
        q_mem: Optional[torch.Tensor] = None,
        k_mem: Optional[torch.Tensor] = None,
        v_mem: Optional[torch.Tensor] = None,
        out_mem: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        # x: (B, N, D)
        B, N, D = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q_spk, q_mem = self.q_lif(q, q_mem)
        k_spk, k_mem = self.k_lif(k, k_mem)
        v_spk, v_mem = self.v_lif(v, v_mem)

        # Reshape to (B, H, N, head_dim)
        q_spk = q_spk.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k_spk = k_spk.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v_spk = v_spk.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Spike matrix multiplication (Sparse Accumulation)
        attn = torch.matmul(q_spk, k_spk.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        out = torch.matmul(attn, v_spk)  # (B, H, N, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        spk_out, out_mem = self.out_lif(out, out_mem)

        return spk_out, (q_mem, k_mem, v_mem, out_mem)


class SpikingTransformerBlock(nn.Module):
    """Spiking Transformer Layer: SpikeAttention + Spiking MLP Feed-Forward Network."""
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, beta: float = 0.9):
        super().__init__()
        self.attn = SpikeAttention(dim=dim, num_heads=num_heads, beta=beta)
        self.ln1 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = nn.Linear(dim, hidden_dim)
        self.mlp_lif1 = LIFNode(beta=beta)
        self.mlp2 = nn.Linear(hidden_dim, dim)
        self.mlp_lif2 = LIFNode(beta=beta)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, states: Optional[dict] = None) -> Tuple[torch.Tensor, dict]:
        states = states or {}
        
        # Self-Attention
        res = x
        x_norm = self.ln1(x)
        attn_out, attn_mems = self.attn(
            x_norm,
            states.get("q_mem"),
            states.get("k_mem"),
            states.get("v_mem"),
            states.get("out_mem"),
        )
        x = res + attn_out

        # MLP
        res = x
        x_norm = self.ln2(x)
        h1 = self.mlp1(x_norm)
        spk_h1, mlp_m1 = self.mlp_lif1(h1, states.get("mlp_m1"))
        h2 = self.mlp2(spk_h1)
        spk_h2, mlp_m2 = self.mlp_lif2(h2, states.get("mlp_m2"))
        x = res + spk_h2

        new_states = {
            "q_mem": attn_mems[0],
            "k_mem": attn_mems[1],
            "v_mem": attn_mems[2],
            "out_mem": attn_mems[3],
            "mlp_m1": mlp_m1,
            "mlp_m2": mlp_m2,
        }
        return x, new_states


class SpikingTransformer(nn.Module):
    """
    Spike-driven Transformer for temporal sequential neuromorphic tasks
    such as Spiking Heidelberg Digits (SHD audio, 700 channels).
    """
    def __init__(
        self,
        input_dim: int = 700,
        num_classes: int = 20,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        num_steps: int = 50,
        beta: float = 0.9,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.input_dim = input_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_lif = LIFNode(beta=beta)

        self.blocks = nn.ModuleList([
            SpikingTransformerBlock(dim=d_model, num_heads=num_heads, beta=beta)
            for _ in range(num_layers)
        ])

        self.head = nn.Linear(d_model, num_classes)
        self.head_lif = LIFNode(beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spike input of shape (batch, num_steps, input_dim) or (num_steps, batch, input_dim)
        """
        if x.dim() == 3 and x.size(1) == self.num_steps:
            x = x.transpose(0, 1)  # (T, B, input_dim)

        T, B, D = x.shape
        proj_mem = None
        block_states = [{} for _ in self.blocks]
        head_mem = None
        out_spikes = []

        for t in range(T):
            xt = x[t]  # (B, input_dim)
            proj = self.input_proj(xt).unsqueeze(1)  # (B, 1, d_model)
            spk_in, proj_mem = self.input_lif(proj, proj_mem)

            token = spk_in
            for i, block in enumerate(self.blocks):
                token, block_states[i] = block(token, block_states[i])

            token_flat = token.squeeze(1)
            logits = self.head(token_flat)
            spk_out, head_mem = self.head_lif(logits, head_mem)
            out_spikes.append(spk_out)

        total_spikes = torch.stack(out_spikes, dim=0).sum(dim=0)
        return total_spikes
