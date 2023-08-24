import torch
import torch.nn as nn
import torch.nn.functional as F

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
    self.num_heads = num_heads

    self.query_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    self.key_value_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=bias) # merge the key and value projections for efficiency reasons

    self.is_causal = is_causal
    if is_causal:
       # note: you can only use this for self attention, when the query and key/value tokens are the same
       # thus the triangle is one half of a square matrix
       self.register_buffer("causal_mask", torch.tril(torch.ones(context_length, context_length)))
    
    self.linear = nn.Linear(embed_dim, embed_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, query, keyvalue, q_token_attn_mask=None, kv_token_attn_mask=None):
    B, T_q, _ = query.size()
    q = self.query_proj(query) # (B,T,embed_dim)
    q = q.view(B, T_q, self.num_heads, self.embed_dim / self.num_heads).transpose(1, 2) # (B,num_heads,T_q,head_size)

    B, T_kv, _ = keyvalue.size()
    k,v = self.key_value_proj(keyvalue).split(self.embed_dim, dim=2) # (B,T,embed_dim) for k and v respectively
    k = k.view(B, T_kv, self.num_heads, self.embed_dim / self.num_heads).transpose(1, 2) # (B,num_heads,T_kv,head_size)
    v = v.view(B, T_kv, self.num_heads, self.embed_dim / self.num_heads).transpose(1, 2) # (B,num_heads,T_kv,head_size)

    # calculate attention matrix
    attn_weights = q @ k.transpose(-2, -1) * (q.shape[-1] ** -0.5) # (B,num_heads,T_q,head_size) @ (B,num_heads,head_size,T_kv) -> (B,num_heads,T_q,T_kv)

    # merge the query and key/value token attention masks
    if q_token_attn_mask is not None and kv_token_attn_mask is not None:
        q_token_attn_mask = q_token_attn_mask.view(B,1,T_q,1) # (B,T_q) -> (B,1,T_q,1)
        kv_token_attn_mask = kv_token_attn_mask.view(B,1,1,T_kv) # (B,T_kv) -> (B,1,1,T_kv)
        attn_mask = q_token_attn_mask & kv_token_attn_mask # (B,1,T_q,1) & (B,1,1,T_kv) -> (B,1,T_q,T_kv)
        attn_weights = attn_weights.masked_fill(attn_mask == 0, -1e9) # (B,num_heads,T_q,T_kv)

    # if applicable, apply the causal mask
    if self.is_causal:
        attn_weights = attn_weights.masked_fill(self.causal_mask[:T_q, :T_kv] == 0, -1e9) # (B,num_heads,T_q,T_kv)
    
    # perform softmax
    attn_weights = F.softmax(attn_weights, dim=-1) # (B,num_heads,T_q,T_kv)

    # use attn_weights to filter info of the value tokens
    outputs = attn_weights @ v # (B,num_heads,T_q,T_kv) @ (B,num_heads,T_kv,head_size) -> (B,num_heads,T_q,head_size)

    # combine the heads
    outputs = outputs.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim) # (B,num_heads,T_q,head_size) -> (B,T_q,embed_dim) 

    return self.dropout(self.linear(outputs))