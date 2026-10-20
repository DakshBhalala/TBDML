"""Neural layers including RMSNorm, BatchEnsembleLinear, and GatedResidualBlock."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., dim)
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class BatchEnsembleLinear(nn.Module):
    """Parameter-efficient BatchEnsemble Linear Layer.

    Maintains a shared weight matrix W and bias b, modulated by k rank-1
    adapters R (input) and S (output) for an ensemble of k submodels.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        bias: bool = True,
        adapter_init_std: float = 0.02,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads

        # Shared base weight matrix
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Rank-1 adapters: initialized with conservative perturbation around 1.0
        # R shape: (k, in_features)
        # S shape: (k, out_features)
        self.R = nn.Parameter(
            torch.ones(num_heads, in_features) + torch.randn(num_heads, in_features) * adapter_init_std
        )
        self.S = nn.Parameter(
            torch.ones(num_heads, out_features) + torch.randn(num_heads, out_features) * adapter_init_std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, in_features) or (k, batch_size, in_features)

        Returns:
            Output tensor of shape (k, batch_size, out_features)
        """
        if x.dim() == 2:
            # Broadcast across k ensemble heads: (k, B, in_features)
            batch_size = x.size(0)
            x_k = x.unsqueeze(0).expand(self.num_heads, batch_size, self.in_features)
        elif x.dim() == 3:
            x_k = x
        else:
            raise ValueError(f"Expected 2D or 3D input tensor, got shape {x.shape}")

        # 1. Modulate inputs with adapter R: (k, B, in_features) * (k, 1, in_features)
        x_mod = x_k * self.R.unsqueeze(1)

        # 2. Linear projection with shared base weight W: (k, B, out_features)
        # Using matmul: (k, B, in_features) @ (in_features, out_features)
        out = torch.matmul(x_mod, self.weight.t())

        # 3. Modulate outputs with adapter S: (k, B, out_features) * (k, 1, out_features)
        out = out * self.S.unsqueeze(1)

        # 4. Add shared bias
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0).unsqueeze(1)

        return out


class GatedResidualBlock(nn.Module):
    """Gated Residual Block with BatchEnsemble and SwiGLU activation."""

    def __init__(
        self,
        hidden_dim: int,
        expansion_factor: float = 2.0,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        intermediate_dim = int(hidden_dim * expansion_factor)

        self.norm = RMSNorm(hidden_dim)
        # Up-projection outputs 2 * intermediate_dim for SwiGLU gating
        self.linear_up = BatchEnsembleLinear(hidden_dim, intermediate_dim * 2, num_heads=num_heads)
        self.linear_down = BatchEnsembleLinear(intermediate_dim, hidden_dim, num_heads=num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (k, batch_size, hidden_dim)

        Returns:
            Tensor of shape (k, batch_size, hidden_dim)
        """
        residual = x
        # Pre-normalization (applies along hidden_dim)
        h = self.norm(x)

        # Up-projection: (k, B, intermediate_dim * 2)
        h = self.linear_up(h)

        # SwiGLU activation
        val, gate = h.chunk(2, dim=-1)
        h = val * F.silu(gate)

        h = self.dropout(h)

        # Down-projection back to hidden_dim: (k, B, hidden_dim)
        h = self.linear_down(h)
        h = self.dropout(h)

        return residual + h
