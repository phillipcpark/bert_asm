import sys

import numpy as np
from sklearn.manifold import TSNE
import torch as th
from torch.utils.data import DataLoader

from dataset import BERTDataset
from arch.bert import BERTLM, BERT 
from utils import load_vocab, is_special_token, replace_masked_tokens

MAX_INSN_LEN = 4
SEQ_LEN      = 2*MAX_INSN_LEN + 3 
BAT_SZ       = 512 #2048
EPOCHS       = 8
USE_GPU      = True

#
if __name__=='__main__':
    ds_path    = sys.argv[1]
    vocab_path = sys.argv[2]
    cp_path    = sys.argv[3]

    x86_vocab = load_vocab(vocab_path)
    bert_ds   = BERTDataset(ds_path, x86_vocab, SEQ_LEN)
    dl        = DataLoader(bert_ds, batch_size=BAT_SZ, num_workers=0)

    bert      = BERT(len(x86_vocab), hidden=64, n_layers=3, attn_heads=4)
    bert.load_state_dict(th.load(cp_path))

    insn_strs = []
    for idx, v in enumerate(dl):
        replace_masked_tokens(v["bert_input"], v["bert_label"])
        embeds = bert.forward(v["bert_input"], v["segment_label"], gen_embed=True).detach().tolist()   

        for insn_idx, insn in enumerate(v["bert_input"]):
            insn_strs.append(x86_vocab.from_seq([t for tok_idx,t in enumerate(insn) if v['segment_label'][insn_idx][tok_idx]==1 \
                                                                                    and not is_special_token(t)], join=True))
        break

    # first 200 insns
    insn_embeds = []
    for e_idx, embed in enumerate(embeds[:200]):
        insn_embeds.append( np.mean([e for _idx,e in enumerate(embed) if v['segment_label'][e_idx][_idx]==1], axis=0) )    

    down_res = TSNE(n_components=2).fit_transform(insn_embeds[:200])

    #FIXME plot
    for e, insn in zip(down_res, insn_strs):
        print(str(e[0]) + "," + str(e[1]) + "," + str(insn))
