import torch
import sentencepiece as spm
import pandas as pd
from torch.utils.data import Dataset
from dataclasses import dataclass

@dataclass
class SPTranslationDatasetConfig:
	src_corpus_file: str
	src_tokenizer_file: str
	target_corpus_file: str
	target_tokenizer_file: str
	max_seq_len: int = 512

class SPTranslationDataset(Dataset):

	UNK_IDX = 0
	BOS_IDX = 1
	EOS_IDX = 2
	PAD_IDX = 3

	def __init__(self, config:SPTranslationDatasetConfig):
		super().__init__()
		self.max_seq_len = config.max_seq_len
		# Read the src and target language corpus files
		with open(config.src_corpus_file, encoding="utf-8") as f:
			src_sentences = f.readlines()
		with open(config.target_corpus_file, encoding="utf-8") as f:
			target_sentences = f.readlines()
		self.data = pd.DataFrame({"src":src_sentences, "target":target_sentences})
		# Load the tokenizers
		self.src_tokenizer = spm.SentencePieceProcessor(model_file=config.src_tokenizer_file)
		self.target_tokenizer = spm.SentencePieceProcessor(model_file=config.target_tokenizer_file)

	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		# Get the src and target sentence
		src_sentence = self.data.iloc[idx]["src"]
		target_sentence = self.data.iloc[idx]["target"]
		# Tokenize the sentences and limit the length to self.max_seq_len
		src_tokens = self.src_tokenizer.encode(src_sentence)[:self.max_seq_len]
		target_tokens = self.target_tokenizer.encode(target_sentence)[:self.max_seq_len]
		# Add the BOS and EOS tokens and convert to tensors
		src_tokens = torch.tensor([SPTranslationDataset.BOS_IDX] + src_tokens + [SPTranslationDataset.EOS_IDX], dtype=torch.long)
		target_tokens = torch.tensor([SPTranslationDataset.BOS_IDX] + target_tokens[:-1] + [SPTranslationDataset.EOS_IDX], dtype=torch.long)
		ground_truth = torch.tensor(target_tokens[1:] + [SPTranslationDataset.EOS_IDX], dtype=torch.long)
		return src_tokens, target_tokens, ground_truth