import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Union
from dataclasses import dataclass
from modules.embedding import SinusoidalPositionalEncoding
from modules.transformer import EncoderBlock, DecoderBlock

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

@dataclass
class TransformerInputBatch:
	encoder_x:torch.Tensor
	decoder_x:torch.Tensor
	encoder_token_attn_mask:torch.Tensor
	decoder_token_attn_mask:torch.Tensor
	ground_truth:torch.Tensor

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

	def forward_encoder(self, encoder_x, encoder_token_attn_mask=None):
		B, encoder_T = encoder_x.size()
		assert encoder_T <= self.config.context_length
		
		encoder_x_tok = self.encoder_token_embedding(encoder_x) # (B,encoder_T) -> (B,encoder_T,embed_dim)
		encoder_x_pos = self.sinusoidal_positional_encoding(torch.arange(encoder_T, device=encoder_x.device)) # (encoder_T,embed_dim)
		encoder_x_pos = torch.repeat_interleave(encoder_x_pos.unsqueeze(0), B, dim=0) # (encoder_T,embed_dim) -> (B,encoder_T,embed_dim)
		encoder_x = self.dropout(encoder_x_tok + encoder_x_pos) # (B,encoder_T,embed_dim)
		for encoder_block in self.encoder_blocks:
			encoder_x = encoder_block(encoder_x, 
			     					  encoder_token_attn_mask=encoder_token_attn_mask) # (B,encoder_T,embed_dim)
		
		return encoder_x # (B,encoder_T,embed_dim)

	def forward_decoder(self, encoder_outputs, decoder_x, encoder_token_attn_mask=None, decoder_token_attn_mask=None):
		B, decoder_T = decoder_x.size()
		assert decoder_T <= self.config.context_length

		decoder_x_tok = self.decoder_token_embedding(decoder_x) # (B,decoder_T) -> (B,decoder_T,embed_dim)
		decoder_x_pos = self.sinusoidal_positional_encoding(torch.arange(decoder_T, device=decoder_x.device)) # (decoder_T,embed_dim)
		decoder_x_pos = torch.repeat_interleave(decoder_x_pos.unsqueeze(0), B, dim=0) # (decoder_T,embed_dim) -> (B,decoder_T,embed_dim)
		decoder_x = self.dropout(decoder_x_tok + decoder_x_pos) # (B,decoder_T,embed_dim)
		for decoder_block in self.decoder_blocks:
			decoder_x = decoder_block(decoder_x, 
			     					  encoder_outputs, 
									  decoder_token_attn_mask=decoder_token_attn_mask, 
									  encoder_token_attn_mask=encoder_token_attn_mask) # (B,decoder_T,embed_dim)
		logits = self.lm_head(decoder_x) # (B,decoder_T,embed_dim) -> (B,decoder_T,decoder_vocab_size)
		return logits

	def forward(self, encoder_x, decoder_x, encoder_token_attn_mask=None, decoder_token_attn_mask=None):
		encoder_outputs = self.forward_encoder(encoder_x, encoder_token_attn_mask=encoder_token_attn_mask) # (B,encoder_T,embed_dim)
		logits = self.forward_decoder(encoder_outputs, 
									  decoder_x, 
									  encoder_token_attn_mask=encoder_token_attn_mask,
									  decoder_token_attn_mask=decoder_token_attn_mask) # (B,decoder_T,decoder_vocab_size)
		return logits
	
	@torch.inference_mode()
	def generate_with_sampling(self, idxs, bos_idx:int, eos_idx:int, temperature:float=1.0, max_new_tokens:int=1000, sampling_strategy:str="multinomial"):
		assert sampling_strategy in ["multinomial", "greedy"], f"Invalid sampling strategy: {sampling_strategy}"
		
		# put model into eval mode
		self.eval()

		# send the input into the model
		encoder_x = idxs.to(self.device).unsqueeze(0) # (T_in) -> (1,T_in)
		encoder_outputs = self.forward_encoder(encoder_x) # (1,T_in,embed_dim)
		decoder_x = torch.tensor([bos_idx], dtype=torch.long, device=self.device).unsqueeze(0) # (1,1)

		# begin sampling
		for _ in range(max_new_tokens):
			logits = self.forward_decoder(encoder_outputs, decoder_x) # (1,T,decoder_vocab_size)
			# cut only to last time step
			logits = logits[:,-1,:]
			# apply temperature
			logits = logits / temperature
			# apply softmax
			probs = F.softmax(logits, dim=-1)
			# sample from the distribution
			if sampling_strategy == "multinomial":
				next_idx = torch.multinomial(probs, num_samples=1) # (1,1)
			elif sampling_strategy == "greedy":
				next_idx = torch.argmax(probs, dim=-1, keepdim=True) # (1,1)
			# add sampled idx to the sequence
			decoder_x = torch.cat([decoder_x, next_idx], dim=-1) # (1,T) -> (1,T+1)
			# yield the generated token idx
			token = int(next_idx[0].cpu().numpy())
			yield token
			# if the token is EOS, stop generating
			if token == eos_idx:
				break 