import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Union
from attention import MultiHeadAttention

class FeedForward(nn.Module):
	def __init__(self, embed_dim:int=512, ffwd_dim:int=2048, activation:Union[str,Callable[[torch.Tensor],torch.Tensor]]=F.relu):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(embed_dim, ffwd_dim),
			activation,
			nn.Linear(ffwd_dim, embed_dim),
		)
	def forward(self, x):
		return self.net(x)
	
class DecoderBlock(nn.Module):
	def __init__(self,
				 context_length:int=512,
				 embed_dim:int=512,
				 num_heads:int=8,
				 bias:bool=False,
				 dropout:float=0.1,
				 ffwd_dim:int=2048,
				 activation:Union[str,Callable[[torch.Tensor],torch.Tensor]]=F.relu,
				 pre_ln:bool=False):
		super().__init__()
		assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
		self.attn1 = MultiHeadAttention(context_length=context_length,
										embed_dim=embed_dim,
										num_heads=num_heads,
										bias=bias,
										dropout=dropout,
										is_causal=True)
		self.ln1 = nn.LayerNorm(embed_dim)
		self.attn2 = MultiHeadAttention(context_length=context_length,
										embed_dim=embed_dim,
										num_heads=num_heads,
										bias=bias,
										dropout=dropout,
										is_causal=False)
		self.ln2 = nn.LayerNorm(embed_dim)
		self.ffwd = FeedForward(embed_dim=embed_dim,
								ffwd_dim=ffwd_dim,
								activation=activation)
		self.ln3 = nn.LayerNorm(embed_dim)
		self.pre_ln = pre_ln

	def forward(self, x, encoder_x, decoder_token_attn_mask=None, encoder_token_attn_mask=None):
		if self.pre_ln:
			x = x + self.attn1(self.ln1(x), self.ln1(x), q_token_attn_mask=decoder_token_attn_mask, kv_token_attn_mask=decoder_token_attn_mask)
			x = x + self.attn2(self.ln2(x), encoder_x, q_token_attn_mask=decoder_token_attn_mask, kv_token_attn_mask=encoder_token_attn_mask)
			x = x + self.ffwd(self.ln3(x))
		else:
			x = self.ln1(x + self.attn1(x, x, q_token_attn_mask=decoder_token_attn_mask, kv_token_attn_mask=decoder_token_attn_mask))
			x = self.ln2(x + self.attn2(x, encoder_x, q_token_attn_mask=decoder_token_attn_mask, kv_token_attn_mask=encoder_token_attn_mask))
			x = self.ln3(x + self.ffwd(x))
		return x

class EncoderBlock(nn.Module):
	def __init__(self,
	      		 context_length:int=512,
				 embed_dim:int=512,
				 num_heads:int=8,
				 bias:bool=False,
				 dropout:float=0.1,
				 ffwd_dim:int=2048,
				 activation:Union[str,Callable[[torch.Tensor],torch.Tensor]]=F.relu,
				 pre_ln:bool=False):
		super().__init__()
		assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
		self.attn1 = MultiHeadAttention(context_length=context_length,
										embed_dim=embed_dim,
										num_heads=num_heads,
										bias=bias,
										dropout=dropout,
										is_causal=False)
		self.ln1 = nn.LayerNorm(embed_dim)
		self.ffwd = FeedForward(embed_dim=embed_dim,
								ffwd_dim=ffwd_dim,
								activation=activation)
		self.ln2 = nn.LayerNorm(embed_dim)
		self.pre_ln = pre_ln

	def forward(self, x, encoder_token_attn_mask=None):
		if self.pre_ln:
			x = x + self.attn1(self.ln1(x), self.ln1(x), q_token_attn_mask=encoder_token_attn_mask, kv_token_attn_mask=encoder_token_attn_mask)
			x = x + self.ffwd(self.ln2(x))
		else:
			x = self.ln1(x + self.attn1(x, x, q_token_attn_mask=encoder_token_attn_mask, kv_token_attn_mask=encoder_token_attn_mask))
			x = self.ln2(x + self.ffwd(x))
		return x