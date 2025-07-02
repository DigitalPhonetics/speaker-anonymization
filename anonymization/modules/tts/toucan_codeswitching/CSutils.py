# based on parts in IMSToucan/Modules/GeneralLayers/ConditionalLayerNorm.py

import torch
from torch import nn


class CSAdaIN1d(nn.Module):
    """
    MIT Licensed

    Copyright (c) 2022 Aaron (Yinghao) Li
    https://github.com/yl4579/StyleTTS/blob/main/models.py
    """

    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        # x: xs (hidden state) (shape: #batch, time, 384
        # s: utt_embedding (shape: #batch, time, 224)
        h = self.fc(s)  # #batch, time, 768
        h = h.view(h.size(0), h.size(1), h.size(2)).transpose(1, 2)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma.transpose(1, 2)) * self.norm(x.transpose(1, 2)).transpose(1, 2) + beta.transpose(1, 2)