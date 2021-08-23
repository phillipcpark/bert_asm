import numpy as np
import sys

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils import load_vocab
from dataset import BERTDataset
from arch.bert import BERTLM, BERT 

MAX_INSN_LEN = 4
SEQ_LEN      = 2*MAX_INSN_LEN + 3 
BAT_SZ       = 512 #2048
EPOCHS       = 8
USE_GPU      = True
MODEL_CP_PATH = 'cps'
  
#
if __name__=='__main__':
    ds_path    = sys.argv[1]
    vocab_path = sys.argv[2]

    x86_vocab = load_vocab(vocab_path)
    bert_ds   = BERTDataset(ds_path, x86_vocab, SEQ_LEN)
    dl        = DataLoader(bert_ds, batch_size=BAT_SZ, num_workers=0)

    bert  = BERT(len(x86_vocab), hidden=64, n_layers=3, attn_heads=4)
    model = BERTLM(bert, len(x86_vocab))
    if (USE_GPU):
        model = model.to('cuda:0')

    optimizer = Adam(model.parameters(), lr=1e-3)

    for e in range(EPOCHS): 
        epoch_loss = []
        for idx, v in enumerate(dl):
            print(str(idx), end= ' ') 

            # FIXME for dev only
            if (idx == 32):
                break
 
            if USE_GPU:
                v["bert_input"]    = v["bert_input"].to('cuda:0')
                v["segment_label"] = v["segment_label"].to('cuda:0')
                v["bert_label"]    = v["bert_label"].to('cuda:0')

            mask_lm_output = model.forward(v["bert_input"], v["segment_label"])

            loss = nn.NLLLoss(ignore_index=0)
            mask_loss = loss(mask_lm_output.transpose(1, 2), v["bert_label"])
            epoch_loss.append(mask_loss.to('cpu').detach())

            optimizer.zero_grad()
            mask_loss.backward()
            optimizer.step()

            if USE_GPU:
                del v["bert_input"]
                del v["segment_label"]
                del v["bert_label"]
                th.cuda.empty_cache()

        ep_loss = np.mean(epoch_loss)
        print(str(ep_loss))
        if (e % 5 == 0):
            th.save(bert.state_dict(), MODEL_CP_PATH + '/ep'+ str(e) + '_loss' + '{:.3f}'.format(ep_loss)) 


