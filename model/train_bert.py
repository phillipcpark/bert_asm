import numpy as np
import sys

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils import parse_args, load_vocab
from dataset import BERTDataset
from arch.bert import BERTLM, BERT 

  
#
if __name__=='__main__':
    cl_args = parse_args()

    ds_path    = cl_args['ds_path']
    vocab_path = cl_args['vocab_path']

    x86_vocab = load_vocab(vocab_path)
    bert_ds   = BERTDataset(ds_path, x86_vocab, cl_args['seq_len'])
    dl        = DataLoader(bert_ds, batch_size=cl_args['bat_sz'], num_workers=0)

    bert  = BERT(len(x86_vocab))
    model = BERTLM(bert, len(x86_vocab))
    if (cl_args['gpu']):
        model = model.to('cuda:0')

    optimizer = Adam(model.parameters(), lr=1e-3)

    for e in range(cl_args['epochs']): 
        epoch_loss = []
        for idx, v in enumerate(dl):
            print(str(idx), end= ' ') 

            # FIXME for dev only
            if (idx == 32):
                break
 
            if cl_args['gpu']:
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

            if cl_args['gpu']:
                del v["bert_input"]
                del v["segment_label"]
                del v["bert_label"]
                th.cuda.empty_cache()

        ep_loss = np.mean(epoch_loss)
        print(str(ep_loss))
        if (e % 5 == 0):
            th.save(bert.state_dict(), cl_args['cpt_dir'] + '/ep'+ str(e) + '_loss' + '{:.3f}'.format(ep_loss)) 


