import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, context_length: int, embed_dim: int):
        super().__init__()

        # FAST VERSION
        pos_encoding = torch.zeros((context_length + 2, embed_dim), requires_grad=False) # add 2 to context length becuase you will add the BOS and EOS tokens
        pos = torch.arange(0, context_length + 2, dtype=torch.float).unsqueeze(1)
        _2i = torch.arange(0, embed_dim, step=2).float()
        pos_encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / embed_dim))) # shape: (context_length + 2, embed_dim // 2)
        pos_encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / embed_dim)))

        # SLOW VERSION (around 8 times slower)
        # for pos in range(context_length + 2):
        #     for i in range(embed_dim // 2):
        #         pos_encoding[pos, 2*i] = torch.sin(torch.tensor(pos / (10000 ** ((2*i)/embed_dim))))
        #         pos_encoding[pos, 2*i+1] = torch.cos(torch.tensor(pos / (10000 ** ((2*i)/embed_dim))))

        self.register_buffer("pos_encoding", pos_encoding)
        
    def forward(self, x):
        return self.pos_encoding[:x.shape(0), :]