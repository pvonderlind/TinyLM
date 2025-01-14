import os.path
from datetime import datetime

import lora
import wandb

import torch
from tqdm import tqdm

import model
from transformers import GPT2TokenizerFast
from torch.utils import data
import numpy as np

"""
Trains a small language model that can learn to speak englisch with very few parameters 
using the TinyStories Dataset.

For Fine-Tuning, you can either use full-weights fine-tuning (just resume training and change the dataset
location) or use low-rank adapters to greatly reduce the amount of trainable parameters (faster fine-tuning).
"""

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def _lazy_file_read(file_obj, chunk_size=1024):
    while True:
        data = file_obj.read(chunk_size)
        if not data:
            break
        yield data


def pre_tokenize_dataset(path, save_path):
    print(f"Running tokenization for {path}")
    try:
        with open(path, 'r', encoding='utf-8') as file:
            arrays = []
            for line in tqdm(_lazy_file_read(file)):
                tokens = tokenizer(line).data['input_ids']
                arrays.append(np.array(tokens, dtype=np.int32))
    except KeyboardInterrupt:
        pass
    finally:
        np.save(save_path, np.concatenate(arrays))
        print(f"Saved tokenized file to binary {save_path}")


class TinyStoriesDataset(data.IterableDataset):

    def __getitem__(self, index):
        pass

    def __init__(self, tokenized_path, block_size: int, device: str = 'cuda'):
        # Each line represents a short story
        self.block_size = block_size
        self.device = device
        self.train = np.load(tokenized_path, mmap_mode='r', allow_pickle=True)

    def __iter__(self):
        while True:
            idx = np.random.randint(0, len(self.train) - self.block_size, 1)[0]
            chunk = self.train[idx:idx + self.block_size + 1]
            source = torch.tensor(chunk[:-1], device=self.device, dtype=torch.long)
            target = torch.tensor(chunk[1:], device=self.device, dtype=torch.long)
            yield source, target

    def __len__(self):
        return len(self.train)


@torch.no_grad()
def eval_model(training_model: torch.nn.Module, val_loader: torch.utils.data.DataLoader):
    training_model.eval()
    losses = torch.zeros(config['EVAL_ITER'])
    for k in range(config['EVAL_ITER']):
        s_val, t_val = next(iter(val_loader))
        val_logits = training_model(s_val)
        val_logits: torch.Tensor = val_logits.view(config['BATCH_SIZE'] * config['BLOCK_SIZE'], config['VOCAB_SIZE'])
        t_val = t_val.view(config['BATCH_SIZE'] * config['BLOCK_SIZE'])
        losses[k] = torch.nn.functional.cross_entropy(val_logits, t_val).item()
    training_model.train()
    return losses.mean()


@torch.no_grad()
def generate_sample_text(training_model: model.TinyLM, max_tokens: int = 200) -> str:
    training_model.eval()
    context = torch.zeros((5, config['BLOCK_SIZE']), dtype=torch.long, device=config['DEVICE'])
    out_tokens = training_model.generate(context, max_new_tokens=max_tokens)
    # Reform to one long piece of text
    out_tokens = out_tokens.view(out_tokens.shape[0] * out_tokens.shape[1])
    training_model.train()
    return tokenizer.decode(out_tokens)


# ----------------------------------------- SETUP ----------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

config = {
    "BLOCK_SIZE": 128,
    "EMB_SIZE": 768,
    "N_ATTENTION_HEADS": 4,
    "N_DECODER_BLOCKS": 2,
    "VOCAB_SIZE": len(tokenizer),
    "MAX_OUT_TOKENS": 200,
    "EVAL_INTERVAL": 1000,
    "EVAL_ITER": 100,
    "LR": 3e-4,
    "BATCH_SIZE": 32,
    "DEVICE": 'cuda' if torch.cuda.is_available() else 'cpu',
    "LOAD_PATH": 'models/tiny_base.pt',
    "SAVE_PATH": 'models/tiny_base_lora.pt',
    "ENABLE_LORA": True,
}
assert config['EMB_SIZE'] % config['N_ATTENTION_HEADS'] == 0
prev_epochs = 0

# Create model instance
print(f"Loading model on device {config['DEVICE']}")
model = model.TinyLM(emb_dim=config['EMB_SIZE'], block_size=config['BLOCK_SIZE'],
                     n_att_heads=config['N_ATTENTION_HEADS'], n_decoders=config['N_DECODER_BLOCKS'],
                     vocab_size=config['VOCAB_SIZE'], device=config['DEVICE'])

model = model.to(config['DEVICE'])

lora_enabled_on_base = False
checkpoint = None
# Load pre-trained base model if it exists
if os.path.exists(config['LOAD_PATH']):
    checkpoint = torch.load(config['LOAD_PATH'], map_location=config['DEVICE'])
    prev_epochs = checkpoint['epoch']
    lora_enabled_on_base = checkpoint['lora_was_enabled']
    model.load_state_dict(checkpoint['model_state_dict'])


# INJECT LoRA, if enabled AND not already enabled in the base weights
# IF lora was in the base weights, there is no need to create the layers again, as they already exist!
# --> Training will just continue to train the LoRA layers.
should_inject_lora = config['ENABLE_LORA'] and not lora_enabled_on_base
if should_inject_lora:
    lora.inject_lora(model, ["self_attention"], 2, 0.1, config['DEVICE'])

# Get trainable parameters
trainable_parameters = [p for p in model.parameters() if p.requires_grad]
total_params = sum(p.numel() for p in trainable_parameters)
print(f"\nTotal trainable parameters: {total_params}")

# Initialize optimizer
optim = torch.optim.Adam(trainable_parameters, lr=config['LR'])
# Load the state of the optimizer only if training is continued with the same structure.
if checkpoint and not should_inject_lora:
    optim.load_state_dict(checkpoint['optimizer_state_dict'])

loss_fn = torch.nn.CrossEntropyLoss()

# DATASET
# ----------------------------------------------------------------------------------------------------------------------
TINY_STORY_TRAIN = 'data/TinyStories-train.txt'
TINY_TOKENIZED = 'data/tiny_tokenized.npy'

if not os.path.exists(TINY_TOKENIZED):
    pre_tokenize_dataset(TINY_STORY_TRAIN, TINY_TOKENIZED)

train_tiny_stories = TinyStoriesDataset(TINY_TOKENIZED, config['BLOCK_SIZE'], device=config['DEVICE'])
train_loader = data.DataLoader(train_tiny_stories, batch_size=config['BATCH_SIZE'])
# ----------------------------------------------------------------------------------------------------------------------
TINY_STORY_VAL = 'data/TinyStories-valid.txt'
TINY_TOKENIZED_VAL = 'data/tiny_tokenized_val.npy'

if not os.path.exists(TINY_TOKENIZED_VAL):
    pre_tokenize_dataset(TINY_STORY_VAL, TINY_TOKENIZED_VAL)

val_tiny_stories = TinyStoriesDataset(TINY_TOKENIZED_VAL, config['BLOCK_SIZE'], device=config['DEVICE'])
val_loader = data.DataLoader(val_tiny_stories, batch_size=config['BATCH_SIZE'])
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- TRAINING -------------------------------------------------------------------

# SETUP Wands & Biases for eval logging
wandb.init(
    project='TinyLM',
    config=config
)
text_table = wandb.Table(columns=['epoch', 'loss', 'predicted text'])

try:
    for b_idx, batch in enumerate(train_loader):
        # Inference
        sources, targets = batch
        logits = model(sources)
        logits = logits.view(config['BATCH_SIZE'] * config['BLOCK_SIZE'], config['VOCAB_SIZE'])
        targets = targets.view(config['BATCH_SIZE'] * config['BLOCK_SIZE'])
        loss = torch.nn.functional.cross_entropy(logits, targets)
        wandb.log({"loss": loss})
        # Weight update
        optim.zero_grad()
        loss.backward()
        optim.step()

        if b_idx % config['EVAL_INTERVAL'] == 0:
            val_loss = eval_model(model, val_loader)
            generated_text = generate_sample_text(model, max_tokens=config['MAX_OUT_TOKENS'])
            print(generated_text)
            wandb.log({"val_loss": val_loss})
except KeyboardInterrupt:
    pass
finally:
    checkpoint_location = config['LOAD_PATH']
    if config['SAVE_PATH']:
        checkpoint_location = config['SAVE_PATH']

    print(f"Saving model to {checkpoint_location} and shutting down training...")
    torch.save({'epoch': prev_epochs + b_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'lora_was_enabled': config['ENABLE_LORA'],
                'config': config,
                }, checkpoint_location)
