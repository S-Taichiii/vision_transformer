import torch
import torch.nn as nn

class MlpHead(nn.Module):
    def __init__(self, emb_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.mlp(z)
        return out