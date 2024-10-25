import torch
import torch.nn as nn

from input_layer import InputLayer
from encoder import Encoder
from mlp_head import MlpHead

class Vit(nn.Module):
    def __init__(self,
                 in_channels: int=3,
                 num_classes: int=10,
                 emb_dim: int=384,
                 num_patch_row: int=2,
                 image_size: int=32,
                 num_blocks: int=7,
                 head: int=8,
                 hidden_dim: int= 384 * 4,
                 dropout: float=0
                 ):

        super().__init__()
        self.input_layer = InputLayer(
            in_channels,
            emb_dim,
            num_patch_row,
            image_size
        )

        self.encoder = Encoder(
            emb_dim = emb_dim,
            head = head,
            hidden_dim = hidden_dim,
            dropout = dropout,
            num_blocks=num_blocks
        )

        self.mlp_head = MlpHead(
            emb_dim = emb_dim,
            num_classes=num_classes
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self. input_layer(z)
        out = self.encoder(out)
        cls_token = out[:, 0]
        pred = self.mlp_head(cls_token)
        return pred

if __name__ == "__main__":
    num_classes = 10
    batch_size, channel, height, width = 2, 3, 32, 32
    x = torch.randn(batch_size, channel, height, width)
    vit = Vit(in_channels=channel, num_classes=num_classes)
    pred = vit(x)

    print(pred.shape)