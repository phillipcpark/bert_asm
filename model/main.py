import csv
import sys
from collections import Counter, OrderedDict

import torch as th
import torch.nn as nn
from torchtext.vocab import vocab
from torch.utils.data import DataLoader
from torch.optim import Adam

import pickle
from sklearn.manifold import TSNE

from dataset import BERTDataset
from bert import BERTLM, BERT 

import numpy as np

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
def load_vocab(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)

#
def is_special_token(token):
    return token <= 4 

#
#
#
if __name__=='__main__':
    ds_path = sys.argv[1]
    ds      = load_ds(ds_path)

    tok_freqs = sorted_tok_freqs(ds) 
    x86_vocab = load_vocab(sys.argv[2])
    seq_len   = 10 #FIXME get_seq_len(ds)

    BAT_SZ  = 4096

    bert_ds = BERTDataset(ds_path, x86_vocab, seq_len)
    dl      = DataLoader(bert_ds, batch_size=BAT_SZ, num_workers=0)

    bert = BERT(len(x86_vocab), hidden=64, n_layers=3, attn_heads=4)
    model = BERTLM(bert, len(x86_vocab))
    optimizer = Adam(model.parameters(), lr=1e-3)

    num_bats = bert_ds.__len__() / BAT_SZ 

    for e in range(8): 
        epoch_loss = []
        for idx, v in enumerate(dl):
            if (idx == 32):
                break

            mask_lm_output = model.forward(v["bert_input"], v["segment_label"])

            loss = nn.NLLLoss(ignore_index=0)
            mask_loss = loss(mask_lm_output.transpose(1, 2), v["bert_label"])

            print(str(idx) + "/" + str(num_bats) + " " + str(mask_loss.detach().tolist()))

            epoch_loss.append(mask_loss.detach())

            optimizer.zero_grad()
            mask_loss.backward()
            optimizer.step()

        print(str(np.mean(epoch_loss)))


    # generate embeddings
    insns = []
    for idx, v in enumerate(dl):

        #FIXME FIXME need instructions without random masking!!
        embeds = bert.forward(v["bert_input"], v["segment_label"], gen_embed=True).detach().tolist()   

        insn_strs = []
        for insn_idx, insn in enumerate(v["bert_input"]):
            insn_strs.append(x86_vocab.from_seq([t for tok_idx,t in enumerate(insn) if v['segment_label'][insn_idx][tok_idx]==1 \
                                                                                    and not is_special_token(t)], join=True))

        # first 100 insns
        insn_embeds = []
        for e_idx, embed in enumerate(embeds[:150]):
            insn_embeds.append( np.mean([e for _idx,e in enumerate(embed) if v['segment_label'][e_idx][_idx]==1], axis=0) )    

        down_res = TSNE(n_components=2).fit_transform(insn_embeds[:150])

        for e, insn in zip(down_res, insn_strs):
            print(str(e[0]) + "," + str(e[1]) + "," + str(insn))

        break





