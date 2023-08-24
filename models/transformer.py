import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Union
from dataclasses import dataclass
from modules.attention import MultiHeadAttention
from modules.embedding import SinusoidalPositionalEncoding

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

@dataclass
class TransformerConfig:
	encoder_vocab_size:int=37000
	decoder_vocab_size:int=37000
	context_length:int=512
	embed_dim:int=512
	num_heads:int=8
	num_encoder_blocks:int=6
	num_decoder_blocks:int=6
	bias:bool=False
	activation:Union[str,Callable[[torch.Tensor],torch.Tensor]]=F.relu
	dropout:float=0.1
	ffwd_dim:int=2048
	pre_ln:bool=False
	weight_tying:bool=False
	use_grad_ckpt:bool=False

class Transformer(nn.Module):
	def __init__(self, config:TransformerConfig):
		super().__init__()
		self.config = config

		# token embeddings
		self.register_buffer("encoder_token_embedding", torch.nn.Embedding(config.encoder_vocab_size, config.embed_dim))
		self.register_buffer("decoder_token_embedding", torch.nn.Embedding(config.decoder_vocab_size, config.embed_dim))
	
		# positional encodings
		self.register_buffer("sinusoidal_positional_encoding", SinusoidalPositionalEncoding(config.context_length, config.embed_dim))

		# dropout after the embeddings
		self.dropout = nn.Dropout(config.dropout)

		# encoder and decoder blocks
		self.encoder_blocks = nn.ModuleList([EncoderBlock(context_length=config.context_length,
														  embed_dim=config.embed_dim,
														  num_heads=config.num_heads,
														  bias=config.bias,
														  dropout=config.dropout,
														  ffwd_dim=config.ffwd_dim,
														  activation=config.activation,
														  pre_ln=config.pre_ln) for _ in range(config.num_encoder_blocks)])
		self.decoder_blocks = nn.ModuleList([DecoderBlock(context_length=config.context_length,
														  embed_dim=config.embed_dim,
														  num_heads=config.num_heads,
														  bias=config.bias,
														  dropout=config.dropout,
														  ffwd_dim=config.ffwd_dim,
														  activation=config.activation,
														  pre_ln=config.pre_ln) for _ in range(config.num_decoder_blocks)])
	
		# final language model head
		self.lm_head = nn.Linear(config.embed_dim, config.decoder_vocab_size, bias=False)

		# weight tying
		if config.weight_tying:
			self.lm_head.weight = self.decoder_token_embedding.weight

	def forward(self, encoder_x, decoder_x, encoder_token_attn_mask, decoder_token_attn_mask):
		B, encoder_T = encoder_x.size()
		_, decoder_T = decoder_x.size()
		assert (encoder_T <= self.config.context_length) and (decoder_T <= self.config.context_length)

		# forward pass through encoder
		encoder_x_tok = self.encoder_token_embedding(encoder_x) # (B,encoder_T) -> (B,encoder_T,embed_dim)
		encoder_x_pos = self.sinusoidal_positional_encoding(torch.arange(encoder_T, device=encoder_x.device)) # (encoder_T,embed_dim)
		encoder_x_pos = torch.repeat_interleave(encoder_x_pos.unsqueeze(0), B, dim=0) # (encoder_T,embed_dim) -> (B,encoder_T,embed_dim)
		encoder_x = self.dropout(encoder_x_tok + encoder_x_pos) # (B,encoder_T,embed_dim)
		for encoder_block in self.encoder_blocks:
			encoder_x = encoder_block(encoder_x, encoder_token_attn_mask) # (B,encoder_T,embed_dim)
		
		# forward pass through decoder
		decoder_x_tok = self.decoder_token_embedding(decoder_x) # (B,decoder_T) -> (B,decoder_T,embed_dim)
		decoder_x_pos = self.sinusoidal_positional_encoding(torch.arange(decoder_T, device=decoder_x.device)) # (decoder_T,embed_dim)
		decoder_x_pos = torch.repeat_interleave(decoder_x_pos.unsqueeze(0), B, dim=0) # (decoder_T,embed_dim) -> (B,decoder_T,embed_dim)
		decoder_x = self.dropout(decoder_x_tok + decoder_x_pos) # (B,decoder_T,embed_dim)
		for decoder_block in self.decoder_blocks:
			decoder_x = decoder_block(decoder_x, encoder_x, decoder_token_attn_mask, encoder_token_attn_mask) # (B,decoder_T,embed_dim)

		# final language model head
		logits = self.lm_head(decoder_x) # (B,decoder_T,embed_dim) -> (B,decoder_T,decoder_vocab_size)
		
		return logits