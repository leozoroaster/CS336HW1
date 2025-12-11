import torch

def data_loading(x, batch_size, context_length, device_str="cpu"):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.long)
    else:
        x = x.to(torch.long).cpu()

    assert x.size(0) > context_length + 1, "Sequence too short for given context_length"

    device = torch.device(device_str)

    starts = torch.randint(0, x.size(0) - context_length - 1, (batch_size,))

    offsets = torch.arange(context_length)  # (context_length,)
    idx = starts.unsqueeze(1) + offsets.unsqueeze(0)  # (B, T)

    input_seqs = x[idx]  # (B, T)
    target_seqs = x[idx + 1]  # (B, T)

    return input_seqs.to(device), target_seqs.to(device)

def save_checkpoint(model,optimizer,epoch,out):
  obj=dict()
  obj["model_weights"]=model.state_dict()
  obj["opt_state"] = optimizer.state_dict()
  obj["epoch"]=epoch
  torch.save(obj,out)

def modify_raw_text(raw_text):
  return "<|endoftext|>".join(raw_text) + "<|endoftext|>"