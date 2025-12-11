import torch
import numpy as np

def data_loading(x, batch_size, context_length, device_str="cpu"):
    device = torch.device(device_str)
    if torch.is_tensor(x):
        # 1D tensor on any device
        x_t = x.to(dtype=torch.long, device="cpu")  # keep on CPU for cheap indexing
        L = x_t.size(0)

        assert L > context_length + 1, "Sequence too short for given context_length"

        starts = torch.randint(0, L - context_length - 1, (batch_size,))
        offsets = torch.arange(context_length)  # (T,)
        idx = starts.unsqueeze(1) + offsets.unsqueeze(0)  # (B, T)

        input_seqs = x_t[idx]          # (B, T)
        target_seqs = x_t[idx + 1]     # (B, T)

        return input_seqs.to(device), target_seqs.to(device)

    else:
        # Assume numpy array / np.memmap
        x_np = np.asarray(x, dtype=np.int64)  # view/cast, does NOT copy memmap
        L = x_np.shape[0]

        assert L > context_length + 1, "Sequence too short for given context_length"

        # sample start positions
        starts = np.random.randint(0, L - context_length - 1, size=batch_size)

        # allocate batch arrays in RAM (small)
        batch_inputs  = np.empty((batch_size, context_length), dtype=np.int64)
        batch_targets = np.empty((batch_size, context_length), dtype=np.int64)

        for i, s in enumerate(starts):
            # slice from memmap: this only reads the needed chunk from disk
            seq = x_np[s : s + context_length + 1]  # length T+1
            batch_inputs[i]  = seq[:-1]
            batch_targets[i] = seq[1:]

        input_seqs = torch.from_numpy(batch_inputs).to(device)
        target_seqs = torch.from_numpy(batch_targets).to(device)

        return input_seqs, target_seqs

def save_checkpoint(model,optimizer,epoch,out):
  obj=dict()
  obj["model_weights"]=model.state_dict()
  obj["opt_state"] = optimizer.state_dict()
  obj["epoch"]=epoch
  torch.save(obj,out)

def modify_raw_text(raw_text):
  return "<|endoftext|>".join(raw_text) + "<|endoftext|>"
