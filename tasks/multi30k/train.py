import torch
import torch.nn.functional as F
import argparse
import tensorboardX
import os
from pathlib import Path

from torch.utils.data import DataLoader
from ...datasets import SPTranslationDataset, SPTranslationDatasetConfig
from ...models import TransformerConfig, Transformer
from ...schedulers import TransformerLRScheduler
from tqdm import tqdm

# ---------------------------------------------------------------------------- #
#                         Ensure Deterministic Behavior                        #
# ---------------------------------------------------------------------------- #

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------- #
#                                Parse Arguments                               #
# ---------------------------------------------------------------------------- #

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--logdir", type=str, default="logs")
parser.add_argument("--random_state", type=int, default=42)
parser.add_argument("--early_stopping_patience", type=int, default=10)
args = parser.parse_args()

EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
WEIGHT_DECAY = args.weight_decay
LOGDIR = args.logdir
RANDOM_STATE = args.random_state
EARLY_STOPPING_PATIENCE = args.early_stopping_patience

# ---------------------------------------------------------------------------- #
#                                 Load Dataset                                 #
# ---------------------------------------------------------------------------- #

train_dataset_config = SPTranslationDatasetConfig(
	src_corpus_file="../../data/multi30k/train/train.en",
	src_tokenizer_file="spm_en.model",
	target_corpus_file="../../data/multi30k/train/train.de",
	target_tokenizer_file="spm_de.model",
	max_seq_len=512
)
train_dataset = SPTranslationDataset(train_dataset_config)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=SPTranslationDataset.collate_fn)

val_dataset_config = SPTranslationDatasetConfig(
	src_corpus_file="../../data/multi30k/val/val.en",
	src_tokenizer_file="spm_en.model",
	target_corpus_file="../../data/multi30k/val/val.de",
	target_tokenizer_file="spm_de.model",
	max_seq_len=512
)
val_dataset = SPTranslationDataset(val_dataset_config)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=SPTranslationDataset.collate_fn)

# ---------------------------------------------------------------------------- #
#                                 Specify Model                                #
# ---------------------------------------------------------------------------- #

model_config = TransformerConfig(
	encoder_vocab_size=16000,
	decoder_vocab_size=16000,
	context_length=512,
	embed_dim=512,
	num_heads=8,
	num_encoder_blocks=6,
	num_decoder_blocks=6,
	bias=False,
	activation=F.relu,
	dropout=0.1,
	ffwd_dim=2048,
	pre_ln=False,
	weight_tying=False,
	use_grad_ckpt=False
)
model = Transformer(model_config)

# ---------------------------------------------------------------------------- #
#                        Specify Optimizer and Scheduler                       #
# ---------------------------------------------------------------------------- #

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=SPTranslationDataset.PAD_IDX)
optimizer = torch.optim.Adam(
	model.parameters(),
	lr=LEARNING_RATE,
    beta1=0.9,
    beta2=0.98,
    eps=1e-9
)
scheduler = TransformerLRScheduler

# ---------------------------------------------------------------------------- #
#                              Main Training Loop                              #
# ---------------------------------------------------------------------------- #

best_val_loss = float("inf")
best_val_loss_counter = 0
best_val_BLEU = -1

for epoch in range(EPOCHS):
	print(f"Epoch {epoch}:")

	# train
	model.train()
	running_train_loss = 0.0
	avg_train_loss = 0.0
	for batch_idx, batch in enumerate(train_pbar := tqdm(train_dataloader)):
		# zero gradients
		optimizer.zero_grad()
		# perform a forward pass
		logits = model(batch.encoder_x, # (B,encoder_T)
					   batch.decoder_x, # (B,decoder_T)
					   encoder_token_attn_mask=batch.encoder_token_attn_mask, # (B,encoder_T)
					   decoder_token_attn_mask=batch.decoder_token_attn_mask) # (B,decoder_T)
		probs = F.softmax(logits, dim=-1) # (B,decoder_T,decoder_vocab_size)
		probs = probs.view(probs.shape[0] * probs.shape[1], probs.shape[2]) # (B*decoder_T,decoder_vocab_size)
		ground_truth = batch.ground_truth # (B,decoder_T)
		ground_truth = ground_truth.view(ground_truth.shape[0] * ground_truth.shape[1]) # (B*decoder_T,)
		# calculate loss
		loss = loss_fn(probs, ground_truth)
		# backward pass
		loss.backward()
		optimizer.step()
		# update running loss
		running_train_loss += loss.item()
		avg_train_loss = running_train_loss / (batch_idx + 1)
		# print progress by updating the tqdm progress bar
		train_pbar.set_description(f"train_loss={avg_train_loss:.5f}")
	
	# validation
	model.eval()
	running_val_loss = 0.0
	avg_val_loss = 0.0
	# full_val_preds = torch.tensor([], dtype=torch.long, device=DEVICE)
	# full_val_ground_truths = torch.tensor([], dtype=torch.long, device=DEVICE)
	for batch_idx, batch in enumerate(val_pbar := tqdm(val_dataloader)):
		logits = model(batch.encoder_x, # (B,encoder_T)
		 			   batch.decoder_x, # (B,decoder_T)
					   encoder_token_attn_mask=batch.encoder_token_attn_mask, # (B,encoder_T)
					   decoder_token_attn_mask=batch.decoder_token_attn_mask) # (B,decoder_T)
		probs = F.softmax(logits, dim=-1) # (B,decoder_T,decoder_vocab_size)
		probs = probs.view(probs.shape[0] * probs.shape[1], probs.shape[2]) # (B*decoder_T,decoder_vocab_size)
		# full_val_preds = torch.cat((full_val_preds, probs), dim=0) # (B,B*decoder_T,decoder_vocab_size)
		ground_truth = batch.ground_truth # (B,decoder_T)
		ground_truth = ground_truth.view(ground_truth.shape[0] * ground_truth.shape[1]) # (B*decoder_T,)
		# full_val_ground_truths = torch.cat((full_val_ground_truths, ground_truth), dim=0) # (B,B*decoder_T)
		# calculate loss
		val_loss = loss_fn(probs, ground_truth)
		# update running loss
		running_val_loss += val_loss.item()
	avg_val_loss = running_val_loss / (batch_idx + 1)
	
	# update learning rate
	scheduler.step()

	# evaluate validation BLEU using whole validation set
	# full_val_preds = full_val_preds.cpu().detach().numpy()
	# full_val_ground_truths = full_val_ground_truths.cpu().detach().numpy()

	# model checkpoint based on validation loss
	if avg_val_loss < best_val_loss:
		print(f"best_val_loss improved from {best_val_loss:.5f} to {avg_val_loss:.5f}, saving model ...")
		best_val_loss = avg_val_loss
		best_val_loss_counter = 0
		payload = {
			"epoch": epoch,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"best_val_loss": best_val_loss,
			# "best_val_BLEU": best_val_BLEU
		}
		torch.save(payload, Path(LOGDIR) / "best_val_loss_model.pth")
	else:
		best_val_loss_counter += 1
		if best_val_loss_counter >= EARLY_STOPPING_PATIENCE:
			print(f"Early stopping as best_val_loss did not improve for {EARLY_STOPPING_PATIENCE} epochs ...")

	# model checkpoint based on validation BLEU

del model
del optimizer
# del full_val_preds
# del full_val_ground_truths
import torch
import torch.nn.functional as F
import argparse
import tensorboardX
import os
from pathlib import Path

from torch.utils.data import DataLoader
from ...datasets import SPTranslationDataset, SPTranslationDatasetConfig
from ...models import TransformerConfig, Transformer
from ...schedulers import TransformerLRScheduler
from tqdm import tqdm

# ---------------------------------------------------------------------------- #
#                         Ensure Deterministic Behavior                        #
# ---------------------------------------------------------------------------- #

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------- #
#                                Parse Arguments                               #
# ---------------------------------------------------------------------------- #

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--logdir", type=str, default="logs")
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
WEIGHT_DECAY = args.weight_decay
LOGDIR = args.logdir
RANDOM_STATE = args.random_state

# ---------------------------------------------------------------------------- #
#                                 Load Dataset                                 #
# ---------------------------------------------------------------------------- #

train_dataset_config = SPTranslationDatasetConfig(
	src_corpus_file="../../data/multi30k/train/train.en",
	src_tokenizer_file="spm_en.model",
	target_corpus_file="../../data/multi30k/train/train.de",
	target_tokenizer_file="spm_de.model",
	max_seq_len=512
)
train_dataset = SPTranslationDataset(train_dataset_config)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=SPTranslationDataset.collate_fn)

val_dataset_config = SPTranslationDatasetConfig(
	src_corpus_file="../../data/multi30k/val/val.en",
	src_tokenizer_file="spm_en.model",
	target_corpus_file="../../data/multi30k/val/val.de",
	target_tokenizer_file="spm_de.model",
	max_seq_len=512
)
val_dataset = SPTranslationDataset(val_dataset_config)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=SPTranslationDataset.collate_fn)

# ---------------------------------------------------------------------------- #
#                                 Specify Model                                #
# ---------------------------------------------------------------------------- #

model_config = TransformerConfig(
	encoder_vocab_size=16000,
	decoder_vocab_size=16000,
	context_length=512,
	embed_dim=512,
	num_heads=8,
	num_encoder_blocks=6,
	num_decoder_blocks=6,
	bias=False,
	activation=F.relu,
	dropout=0.1,
	ffwd_dim=2048,
	pre_ln=False,
	weight_tying=False,
	use_grad_ckpt=False
)
model = Transformer(model_config)

# ---------------------------------------------------------------------------- #
#                        Specify Optimizer and Scheduler                       #
# ---------------------------------------------------------------------------- #

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=SPTranslationDataset.PAD_IDX)
optimizer = torch.optim.Adam(
	model.parameters(),
	lr=LEARNING_RATE,
    beta1=0.9,
    beta2=0.98,
    eps=1e-9
)
scheduler = TransformerLRScheduler

# ---------------------------------------------------------------------------- #
#                              Main Training Loop                              #
# ---------------------------------------------------------------------------- #

best_val_loss = float("inf")
best_val_loss_counter = 0
best_val_BLEU = -1

for epoch in range(EPOCHS):
	print(f"Epoch {epoch}:")

	# train
	model.train()
	running_train_loss = 0.0
	avg_train_loss = 0.0
	for batch_idx, batch in enumerate(train_pbar := tqdm(train_dataloader)):
		# zero gradients
		optimizer.zero_grad()
		# perform a forward pass
		logits = model(batch.encoder_x, # (B,encoder_T)
					   batch.decoder_x, # (B,decoder_T)
					   encoder_token_attn_mask=batch.encoder_token_attn_mask, # (B,encoder_T)
					   decoder_token_attn_mask=batch.decoder_token_attn_mask) # (B,decoder_T)
		probs = F.softmax(logits, dim=-1) # (B,decoder_T,decoder_vocab_size)
		probs = probs.view(probs.shape[0] * probs.shape[1], probs.shape[2]) # (B*decoder_T,decoder_vocab_size)
		ground_truth = batch.ground_truth # (B,decoder_T)
		ground_truth = ground_truth.view(ground_truth.shape[0] * ground_truth.shape[1]) # (B*decoder_T,)
		# calculate loss
		loss = loss_fn(probs, ground_truth)
		# backward pass
		loss.backward()
		optimizer.step()
		# update running loss
		running_train_loss += loss.item()
		avg_train_loss = running_train_loss / (batch_idx + 1)
		# print progress by updating the tqdm progress bar
		train_pbar.set_description(f"train_loss={avg_train_loss:.5f}")
	
	# validation
	model.eval()
	running_val_loss = 0.0
	avg_val_loss = 0.0
	full_val_preds = torch.tensor([], dtype=torch.long, device=DEVICE)
	full_val_ground_truths = torch.tensor([], dtype=torch.long, device=DEVICE)
	for batch_idx, batch in enumerate(val_pbar := tqdm(val_dataloader)):
		logits = model(batch.encoder_x, # (B,encoder_T)
		 			   batch.decoder_x, # (B,decoder_T)
					   encoder_token_attn_mask=batch.encoder_token_attn_mask, # (B,encoder_T)
					   decoder_token_attn_mask=batch.decoder_token_attn_mask) # (B,decoder_T)
		probs = F.softmax(logits, dim=-1) # (B,decoder_T,decoder_vocab_size)
		probs = probs.view(probs.shape[0] * probs.shape[1], probs.shape[2]) # (B*decoder_T,decoder_vocab_size)
		full_val_preds = torch.cat((full_val_preds, probs), dim=0) # (B,B*decoder_T,decoder_vocab_size)
		ground_truth = batch.ground_truth # (B,decoder_T)
		ground_truth = ground_truth.view(ground_truth.shape[0] * ground_truth.shape[1]) # (B*decoder_T,)
		full_val_ground_truths = torch.cat((full_val_ground_truths, ground_truth), dim=0) # (B,B*decoder_T)
		# calculate loss
		val_loss = loss_fn(probs, ground_truth)
		# update running loss
		running_val_loss += val_loss.item()
	avg_val_loss = running_val_loss / (batch_idx + 1)
	
	# update learning rate
	scheduler.step()

	# evaluate validation BLEU using whole validation set
	full_val_preds = full_val_preds.cpu().detach().numpy()
	full_val_ground_truths = full_val_ground_truths.cpu().detach().numpy()
	# blah blah evaluate BLEU

	# model checkpoint based on validation loss

	# model checkpoint based on validation BLEU