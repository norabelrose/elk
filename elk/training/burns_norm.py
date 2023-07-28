import torch
from torch import Tensor, nn


class BurnsNorm(nn.Module):
    """Burns et al. style normalization. Minimal changes from the original code."""
    scale: bool = True

    def forward(self, x: Tensor) -> Tensor:
        x_normalized: Tensor = x - x.mean(dim=0)

        std = torch.linalg.norm(x_normalized, dim=0) / torch.sqrt(
            torch.tensor(x_normalized.shape[0], dtype=torch.float32)
        )
        assert std.dim() == x.dim() - 1

        # Compute the dimensions over which we want to compute the mean standard deviation
        dims = tuple(range(1, std.dim())) # exclude the first dimension (v)

        avg_norm = std.mean(dim=dims)

        # Add a singleton dimension at the beginning to allow broadcasting. 
        # This compensates for the dimension we lost when computing the norm.
        avg_norm = avg_norm.unsqueeze(0)

        # Add singleton dimensions at the end to allow broadcasting.
        # This compensates for the dimensions we lost when computing the mean.
        for _ in range(1, x.dim() - 1):
            avg_norm = avg_norm.unsqueeze(-1)

        return x_normalized / avg_norm if self.scale else x_normalized