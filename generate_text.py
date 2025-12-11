import transfomer
import optimizer
import BPE_tokenizer
import torch
import numpy as np

def sample(v: torch.Tensor):
    v = v.cpu().numpy()
    p = np.random.rand()
    s = 0.0
    for i in range(len(v)):
        s += float(v[i])
        if s >= p:
            return i
    return len(v) - 1

def decode_prompt(LM,tokenizer,prompts,max_len=10,temperature=1,p=None):
  output_answers=[]
  for prompt in prompts:
    encoded_prompt = tokenizer.encode(prompt)
    generated_tokens=list(encoded_prompt)
    L=len(generated_tokens)
    for _ in range(max_len):
      curr_prompt=generated_tokens
      input_tensor = torch.tensor(curr_prompt, dtype=torch.long, device=device).unsqueeze(0)
      with torch.no_grad():
            logits = LM(input_tensor)
      next_token_raw=logits[0,-1,:]

      next_token_prob=transfomer.softmax(next_token_raw/temperature)

      if p is not None:
        sorted_probs, sorted_indices = torch.sort(next_token_prob, descending=True)
        S = 0.0
        threshold = 0.0
        for i in range(len(sorted_probs)):
            S += float(sorted_probs[i])
            if S >= p:
                threshold = float(sorted_probs[i])
                break

        for i in range(len(next_token_prob)):
          if next_token_prob[i]<threshold:
            next_token_prob[i]=0

        next_token_prob=next_token_prob/next_token_prob.sum()

      next_token=sample(next_token_prob)

      generated_tokens.append(next_token)

      if tokenizer.decode([next_token])=="<|endoftext|>":
        break
    output_answers.append(tokenizer.decode(generated_tokens))
  return output_answers

checkpoint_path = 'train_LM_sanity_check.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the checkpoint dictionary
checkpoint = torch.load(checkpoint_path, map_location=device)

LM=transfomer.transformer_lm(128,4,256,10000,256,2,10000).to(device)

OPT=optimizer.AdamW(LM.parameters(), decay=1e-4, lr=1e-3)

# Load model weights
LM.load_state_dict(checkpoint['model_weights'])

# Load optimizer state
OPT.load_state_dict(checkpoint['opt_state'])

print(f"Model and Optimizer loaded from epoch {checkpoint['epoch']}")

LM.eval()

vocab ,merges=BPE_tokenizer.train_bpe(r"TinyStories.txt",10000,["<|endoftext|>"])

tokenizer=BPE_tokenizer.bpe_tokenizer(vocab,merges, ["<|endoftext|>"])

prompts=["The capital of France is", "The history of the United States began", "A computer program is"]

print(decode_prompt(LM, tokenizer,prompts))