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
        self.use_mask = use_mask
        if use_mask:
            self.register_buffer('triangle_mask', torch.tril(torch.ones(context_length, context_length)))
        self.is_cross_attention = is_cross_attention

    def forward(self, x, attn_mask=None, encoder_x=None):
        # x and encoder_x both have shape (B,T,embed_dim)
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
        # q,k,v will have shape (B,T,head_size)

        # calculate attention matrix
        attn_weights = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5) # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)
        
        # use attn_mask to mask out tokens (such as padding tokens)
        if attn_mask is not None:
        

        if self.use_mask:
            num_timesteps = q.shape[-2]
            attn_weights = attn_weights.masked_fill(self.triangle_mask[:num_timesteps, :num_timesteps] == 0, float('-inf')) # (B,T,T)
        attn_weights = F.softmax(attn_weights, dim=-1) # (B,T,T)
        output = attn_weights @ v # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return output

class MultiHeadAttention(nn.Module):
  def __init__(self,
               context_length:int=512,
               embed_dim:int=512,
               num_heads:int=8,
               bias:bool=False,
               dropout:float=0.1,
               is_causal:bool=False):
    super().__init__()
    assert embed_dim % num_heads == 0
    self.embed_dim = embed_dim
    self.head_size = embed_dim / num_heads

    self.query_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    self.key_value_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=bias) # merge the key and value projections for efficiency reasons

    if is_causal:
       # note: you can only use this for self attention
       self.register_buffer("causal_mask", torch.tril(torch.ones(context_length, context_length)))
    
    self.linear = nn.Linear(embed_dim, embed_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, query_x, key_value_x, attn_mask=None):
    
    B, T_q, C = query_x.size()
    q = self.query_proj(query_x) # (B,T,embed_dim)

    
    k,v = self.key_value_proj(key_value_x).split(self.embed_dim, dim=2) # (B,T,embed_dim) for k and v respectively

    x = torch.cat([h(x, attn_mask=attn_mask, encoder_x=encoder_x) for h in self.heads], dim=-1)
    x = self.linear(x)
    x = self.dropout(x)
    return x