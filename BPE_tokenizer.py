
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path:str,vocab_size:int,special_tokens:list[str]):

  #initialize vocabulary
  vocab_dict = {}

  for i in range(256):
      vocab_dict[i] = bytes([i])

  for token in special_tokens:
      vocab_dict[len(vocab_dict)] = token.encode("utf-8")

  #read file
  raw_text=open(input_path,'r', encoding="utf-8").read()

  #pre tokenize
  raw_text=re.findall(PAT, raw_text)

  #first pass
  word_dict=dict()
  for word in raw_text:
    word=tuple(word.encode("utf-8"))
    if word in word_dict:
      word_dict[word]+=1
    else:
      word_dict[word]=1

  #merge
  merges=[]
  while len(vocab_dict) < vocab_size:
    max_count=0
    pair_count=dict()
    best_pair=None
    best_key=None
    for word,count in word_dict.items():
      for i in range(len(word)-1):
        pair=(word[i],word[i+1])
        merged=vocab_dict[word[i]]+vocab_dict[word[i+1]]
        if pair in pair_count:
          pair_count[pair]+=count
        else:
          pair_count[pair]=count
        if pair_count[pair] > max_count or (pair_count[pair] == max_count and merged < best_key):
          max_count = pair_count[pair]
          best_pair=pair
          best_key=merged
    if max_count<2:
      break
    else:
      merges.append((vocab_dict[best_pair[0]],vocab_dict[best_pair[1]]))
      vocab_dict[len(vocab_dict)] = best_key
      new_word_dict=dict()
      for word,count in word_dict.items():
        new_word=[]
        i=0
        word_len=len(word)
        if word_len<2:
          new_word_dict[word] = new_word_dict.get(word, 0) + count
          continue
        total_len=len(vocab_dict)-1
        while i<word_len:
          curr_id=word[i]
          if i==word_len-1:
            new_word.append(curr_id)
            i+=1
          else:
            next_id=word[i+1]
            if curr_id==best_pair[0] and next_id==best_pair[1]:
              new_word.append(total_len)
              i+=2
            else:
              new_word.append(curr_id)
              i+=1
        new_word=tuple(new_word)
        if new_word in new_word_dict:
          new_word_dict[new_word] += count
        else:
          new_word_dict[new_word] = count
      word_dict = new_word_dict
  return vocab_dict, merges

class bpe_tokenizer:
  def __init__(self,vocab,merges,special_tokens=None):
    self.vocab=vocab
    self.merges=merges
    self.special_tokens=special_tokens
    self.vocab_inverse = {v: k for k, v in self.vocab.items()}
    self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    self.vocab_len = len(self.vocab)

    if self.special_tokens is not None:
      special_tokens_bytes = set([token.encode("utf-8") for token in self.special_tokens])

      for byte_token in special_tokens_bytes:
        if byte_token not in self.vocab.values():
          self.vocab[self.vocab_len] = byte_token
          self.vocab_len += 1

  @classmethod
  def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
    vocab_dict = dict()
    with open(vocab_filepath, 'r', encoding='utf-8') as f:
      for line in f:
        curr_combo = line.strip().split('\t')
        if len(curr_combo) == 2:
          curr_id = int(curr_combo[0])
          curr_bytes = bytes.fromhex(curr_combo[1])
          vocab_dict[curr_id] = curr_bytes

    merges = []
    with open(merges_filepath, 'r', encoding='utf-8') as f:
      for line in f:
        curr_combo = line.strip().split('\t')
        if len(curr_combo) == 2:
          curr_bytes_1 = bytes.fromhex(curr_combo[0])
          curr_bytes_2 = bytes.fromhex(curr_combo[1])
          merges.append((curr_bytes_1, curr_bytes_2))

    return cls(vocab_dict, merges, special_tokens)

  def get_pairs(self, symbols):
    """Return set of adjacent symbol pairs in the current word."""
    pairs = set()
    for i in range(len(symbols) - 1):
      pairs.add((symbols[i], symbols[i + 1]))
    return pairs

  def encode(self, text: str) -> list[int]:
    # split text into tokens/words
    tokens = re.findall(PAT, text)
    output_ids = []

    for word in tokens:
      # start from bytes
      word_bytes = word.encode("utf-8")
      # list of byte tokens: [b'a', b'b', ...]
      symbols = [bytes([b]) for b in word_bytes]

      if not symbols:
        continue

      pairs = self.get_pairs(symbols)

      while True:
        # find best (lowest-rank) pair to merge
        best_pair = None
        best_rank = None
        for pair in pairs:
          rank = self.merge_ranks.get(pair)
          if rank is not None and (best_rank is None or rank < best_rank):
            best_rank = rank
            best_pair = pair

        # no pair can be merged → done
        if best_pair is None:
          break

        first, second = best_pair
        new_symbols = []
        i = 0
        while i < len(symbols):
          # if we see the pair, merge it
          if (
            i < len(symbols) - 1
            and symbols[i] == first
            and symbols[i + 1] == second
          ):
            new_symbols.append(first + second)
            i += 2
          else:
            new_symbols.append(symbols[i])
            i += 1

        symbols = new_symbols
        if len(symbols) == 1:
          break
        pairs = self.get_pairs(symbols)

      # map merged byte symbols to token IDs
      for b in symbols:
        output_ids.append(self.vocab_inverse[b])

    return output_ids

  def encode_iterable(self,iterable):
    for chunk in iterable:
      for token_id in self.encode(chunk):
        yield token_id

  def decode(self, ids: list[int]) -> str:
    chunks = []
    for num in ids:
      chunks.append(self.vocab[num])
    return b"".join(chunks).decode("utf-8")