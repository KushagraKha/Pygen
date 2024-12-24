import torch
from model import LightweightTransformer

vocab_size = 32000  # Must match your tokenizer's vocab size
d_model = 256
nhead = 4
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 512

# 2. Create an instance
model = LightweightTransformer(
    vocab_size=vocab_size,
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward
)

# 3. Load state_dict
model_path = "my_lightweight_transformer.pth"
state_dict = torch.load(model_path, map_location=torch.device("cpu"))  # or "cuda"
model.load_state_dict(state_dict)
model.eval()
print("Model loaded and set to eval mode.")

def generate_code(model, tokenizer, prompt, max_length=128):
    model.eval()
    
    # Tokenize the prompt (src)
    src_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Start decoding with [BOS] (CodeT5 may use <s> or <extra_id_0>)
    # We can just use the tokenizer's special_bos_token_id if it exists
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        # If CodeT5 doesn't define bos_token_id, you might use <s> or just a known ID
        bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
    
    generated = torch.tensor([[bos_token_id]], dtype=torch.long).to(model.device)
    
    for step in range(max_length):
        # Forward pass
        logits = model(src_ids, generated)  # [batch_size=1, seq_len, vocab_size]
        
        # Take the last token logits
        next_token_logits = logits[:, -1, :]  # shape: [1, vocab_size]
        
        # Greedy select the next token
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)  # [1,1]
        
        # Append next_token to generated sequence
        generated = torch.cat([generated, next_token], dim=1)
        
        # If it's the EOS token, break
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    # Decode the generated token IDs (excluding the initial BOS token)
    generated_text = tokenizer.decode(generated[0, 1:], skip_special_tokens=True)
    return generated_text

# Example usage:
prompt_text = "Write a Python function that checks if a number is prime."
prediction = generate_code(model, tokenizer, prompt_text)
print("Generated Code:\n", prediction)