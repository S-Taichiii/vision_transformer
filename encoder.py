import torch
import torch.nn as nn
import torch.nn.functional as F
from input_layer import InputLayer

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 emb_dim:int = 384,
                 head:int = 3,
                 dropout: float = 0):

        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // head
        self.sqrt_dh = self.head_dim ** 0.5

        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)

        self.w_o = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, num_patch, _ = z.size()

        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k_T = k.transpose(2, 3)
        dots = (q @ k_T) / self.sqrt_dh
        attn = F.softmax(dots, dim = 1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, num_patch, self.emb_dim)

        out = self.w_o(out)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, 
                 emb_dim: int=384,
                 head: int=3,
                 hidden_dim: int=384* 4,
                 dropout: float=0):
        
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.mhsa = MultiHeadSelfAttention(
            emb_dim=emb_dim,
            head=head,
            dropout=dropout
        )
        self.ln2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.mhsa(self.ln1(z)) + z
        out = self.mlp(self.ln2(out)) + out
        return out

class Encoder(nn.Module):
    def __init__(self,
                 emb_dim: int=384,
                 head: int=3,
                 hidden_dim: int=384 * 4,
                 dropout: float=0,
                 num_blocks: int=3):
        
        super().__init__()
        self.encoder = nn.Sequential(*[
            EncoderBlock(
                emb_dim = emb_dim,
                head = head,
                hidden_dim = hidden_dim,
                dropout = dropout,
            )
            for _ in range(num_blocks)])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.encoder(z)
        return out

