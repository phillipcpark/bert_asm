import csv
import sys
from collections import Counter, OrderedDict

import torch as th
import torch.nn as nn
from torchtext.vocab import vocab

def load_ds(path):
    asm_tokens = []
    with open(path, 'r') as fh:
        reader = csv.reader(fh, delimiter=',')

        curr_bb = []
        for l in reader:
            if not(l==[]):
                curr_bb.append(l)
            else:
                asm_tokens.append(curr_bb)
                curr_bb = []
        asm_tokens.append(curr_bb)
    return asm_tokens

#
def sorted_tok_freqs(bbs: list) -> dict:
    all_toks = [tok for bb in ds for insn in bb for tok in insn] 
    tok_counts = Counter(all_toks)

    tok_counts = sorted(tok_counts.items(), key=lambda x: x[1], reverse=True)
    tok_counts = OrderedDict(tok_counts)
    return tok_counts 

#
#
#
if __name__=='__main__':
    ds_path = sys.argv[1]
    ds      = load_ds(ds_path)

    tok_freqs = sorted_tok_freqs(ds) 

    x86_vocab = vocab(tok_freqs, min_freq=8)
    x86_vocab.append_token('unk')
    vocab_sz = x86_vocab.__len__()
    x86_vocab.set_default_index(vocab_sz-1)
    embeder = nn.Embedding(vocab_sz, 32)


