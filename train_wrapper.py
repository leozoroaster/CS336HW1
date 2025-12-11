import BPE_tokenizer
import transfomer
import optimizer
import data_process
import torch
from pathlib import Path
import numpy as np

from datasets import load_dataset

def train_model_TinyStories(d_model=512,h=16,d_ff=1344,vocab_size=10000,context_length=256,num_layers=4,theta=10000,raw_lr=1e-3,decay=1e-4,epoch_num=20,batch_num=256,batch_size=64,data_dir="data", save_dir="checkpoints",device=None):
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("start dataset loading")
    dataset=load_dataset("roneneldan/TinyStories")
    print("dataset loading successful")

    print("start text saving")
    text=dataset["train"]["text"]
    text_path = data_dir / "TinyStories.txt"
    token_path = data_dir / "TinyStories_tokens.int32"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text))
    print("text saving successful")

    vocab_file = data_dir / "TinyStories_vocab.txt"
    merges_file = data_dir / "TinyStories_merges.txt"

    #one-time training of BPE
    #print("start training BPE")
    #bpe_text=dataset["train"].select(range(100_000))["text"]
    #bpe_text_path = data_dir / "TinyStories_100k.txt"
    #with open(bpe_text_path, "w", encoding="utf-8") as f:
        #f.write("\n".join(bpe_text))
    #print("text saving successful")
    #vocab, merges = BPE_tokenizer.train_bpe(str(bpe_text_path), vocab_size, ["<|endoftext|>"])
    #print("training BPE successful")

    #BPE_tokenizer.bpe_tokenizer.save_tokenizer(vocab, merges, vocab_file, merges_file)
    #print(f"saved tokenizer to {vocab_file} and {merges_file}")
    #end training

    print("start tokenizing")
    tokenizer=BPE_tokenizer.bpe_tokenizer.from_files(
        vocab_filepath=str(vocab_file),
        merges_filepath=str(merges_file),
        special_tokens=["<|endoftext|>"],
    )
    with open(text_path, "r", encoding="utf-8") as fin, \
            open(token_path, "wb") as fout:

        for line in fin:
            line = line.rstrip("\n")

            raw_text = data_process.modify_raw_text(line)
            token_ids = tokenizer.encode(raw_text)

            np.array(token_ids, dtype=np.int32).tofile(fout)

            end_id = tokenizer.encode("<|endoftext|>")[0]
            np.array([end_id], dtype=np.int32).tofile(fout)

    raw_tokens = np.memmap(
        token_path,
        dtype=np.int32,
        mode="r"
    )
    print("tokenizing successful")
    print("num of tokens ", len(raw_tokens))
    print("tokens per epoch", batch_num*batch_size*context_length)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = device
    print("Using device:", device)
    epoch_num=epoch_num
    batch_num=batch_num

    print("creating LM")
    LM=transfomer.transformer_lm(d_model,h,d_ff,vocab_size,context_length,num_layers,theta).to(device)
    print("LM created")

    print("creating OPT")
    OPT=optimizer.AdamW(LM.parameters(), decay=decay, lr=raw_lr)
    print("OPT created")

    print("start training")
    LM.train()

    for epoch in range(epoch_num):
        print("epoch ", epoch + 1)
        epoch_loss = 0.0
        for batch in range(batch_num):
            train_input, train_pred = data_process.data_loading(raw_tokens, batch_size, context_length, device_str=str(device))
            x = train_input.to(device)
            y = train_pred.to(device)
            OPT.zero_grad()
            logits = LM(x)
            loss_per_token = optimizer.cross_entropy(logits, y)
            loss = loss_per_token.mean()
            epoch_loss += loss.item()

            loss.backward()

            OPT.step(scheduler=(epoch,raw_lr,raw_lr*0.01,int(0.15*epoch_num),int(0.95*epoch_num)))
        print("epoch avg loss ", epoch_loss / batch_num)
    print("finished training")
    data_process.save_checkpoint(LM, OPT, epoch_num, save_dir / "train_LM_sanity_check.pt")
    print("finished model saving")

if __name__ == "__main__":
    train_model_TinyStories()

