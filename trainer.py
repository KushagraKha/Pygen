from torch.utils.data import DataLoader
import os
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from APPSCollator import APPSCollator
from DatasetClass import APPSDataset
from transformers import AutoTokenizer
from model import LightweightTransformer

MODEL_NAME = "Salesforce/codet5-base"  # or any other CodeT5 variant
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Vocabulary size needed for your custom Transformer
vocab_size = len(tokenizer)
print("CodeT5 vocabulary size:", vocab_size)


# Hyperparams
batch_size = 2
epochs = 2
lr = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate dataset & collator
train_dataset = APPSDataset(
    questions_dir="train-questions",
    solutions_dir="train-solutions",
    max_samples=None  # or a small int (e.g., 100) for debugging
)

collator = APPSCollator(tokenizer, max_source_length=256, max_target_length=256)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

# Instantiate the model
model = LightweightTransformer(
    vocab_size=vocab_size,
    d_model=256,             # smaller dimension for a "lightweight" approach
    nhead=4,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=512
).to(device)

print("BOS token ID:", tokenizer.bos_token_id)
print("EOS token ID:", tokenizer.eos_token_id)
print("PAD token ID:", tokenizer.pad_token_id)


# Define optimizer & loss
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=-100)  # ignore padding in the label

# Simple training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)  # not used in our custom model but available
        labels = batch["labels"].to(device)

        tgt_input = labels.clone()
        pad_id = tokenizer.pad_token_id
        tgt_input[tgt_input == -100] = pad_id

        # Our model expects src, tgt
        # In seq2seq, typically we shift the labels by 1. For simplicity, let's just pass labels as is.
        # We'll assume the model is trained to predict labels[i+1] from input[i], etc.
        logits = model(input_ids, tgt_input)  # Note: 'labels' used as 'tgt_input' here for teacher forcing

        # Reshape for cross-entropy
        # logits: [batch_size, tgt_seq_len, vocab_size]
        # labels: [batch_size, tgt_seq_len]
        logits_reshaped = logits.view(-1, vocab_size)
        labels_reshaped = labels.view(-1)

        loss = criterion(logits_reshaped, labels_reshaped)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / (step + 1)
    print(f"Epoch [{epoch+1}/{epochs}] finished with average loss {avg_loss:.4f}")

model_save_path = "my_lightweight_transformer.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")