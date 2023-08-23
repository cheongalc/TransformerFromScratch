import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, 
                 context_length:int=512,
                 embed_dim:int=512,
                 head_size:int=64, # because original paper uses 512 hidden size and 8 heads
                 bias:bool=False,
                 dropout:float=0.1,
                 use_mask:bool=False,
                 is_cross_attention:bool=False):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(embed_dim, head_size, bias=bias)
        self.key = nn.Linear(embed_dim, head_size, bias=bias)
        self.value = nn.Linear(embed_dim, head_size, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.use_mask = use_mask
        if use_mask:
            self.register_buffer('triangle_mask', torch.tril(torch.ones(context_length, context_length)))
        self.is_cross_attention = is_cross_attention

    def forward(self, x, attn_mask=None, encoder_x=None):
        # x,encoder_x both have shape (B,T,embed_dim)
        if self.is_cross_attention:
            # for cross attention, q is from the decoder, while k & v are from the encoder
            q = self.query(x)
            k = self.key(encoder_x)
            v = self.key(encoder_x)
        else:
            # for self attention, q,k,v are all from the same source
            q = self.query(x)
            k = self.key(x)
            v = self.value(x)
        # q,k,v all have shape (B,T,head_size)
        attn_weights = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5) # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)
        if self.use_mask: # means that we are in the decoder
            num_timesteps = q.shape[-2]
            attn_weights = attn_weights.masked_fill(self.triangle_mask[:num_timesteps, :num_timesteps] == 0, float('-inf')) # (B,T,T)
        attn_weights = F.softmax(attn_weights, dim=-1) # (B,T,T)
        attn_weights = self.dropout(attn_weights)
        output = attn_weights @ v # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return output

class MultiHeadAttention(nn.Module):
  def __init__(self,
               context_length:int=512,
               embed_dim:int=512,
               num_heads:int=8,
               bias:bool=False,
               dropout:float=0.1,
               use_mask:bool=False,
               is_cross_attention:bool=False):
    super().__init__()
    self.heads = nn.ModuleList([Head(context_length=context_length,
                                     embed_dim=embed_dim,
                                     head_size=embed_dim // num_heads,
                                     bias=bias,
                                     dropout=dropout,
                                     use_mask=use_mask,
                                     is_cross_attention=is_cross_attention) for _ in num_heads])
    self.proj = nn.Linear(embed_dim, embed_dim) # this is there to provide some "thinking time" after the attention
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, attn_mask=None, encoder_x=None):
    x = torch.cat([h(x, attn_mask=attn_mask, encoder_x=encoder_x) for h in self.heads], dim=-1)
    x = self.proj(x)
    x = self.dropout(x)
    return x