import sys

import numpy as np
from sklearn.manifold import TSNE
import torch as th
from torch.utils.data import DataLoader

from dataset import BERTDataset
from arch.bert import BERTLM, BERT 
import utils

#
# generate embeddings using pretrained x86-64 BERT, perform dimensionality reduction via T-SNE, and write scatter plot
#
if __name__=='__main__':
    cl_args = utils.parse_args(for_train=False)

    ds_path    = cl_args['ds_path']
    vocab_path = cl_args['vocab_path']
    cp_path    = cl_args['model_path']

    x86_vocab = utils.load_vocab(vocab_path)
    bert_ds   = BERTDataset(ds_path, x86_vocab, cl_args['seq_len'])
    dl        = DataLoader(bert_ds, batch_size=cl_args['bat_sz'], num_workers=0)

    bert      = BERT(len(x86_vocab))
    bert.load_state_dict(th.load(cp_path))

    embeds    = None
    insn_strs = []

    for idx, v in enumerate(dl):
        utils.replace_masked_tokens(v["bert_input"], v["bert_label"])
        embeds = bert.forward(v["bert_input"], v["segment_label"], gen_embed=True).detach().tolist()   

        for insn_idx, insn in enumerate(v["bert_input"]):
            insn_strs.append(x86_vocab.from_seq([t for tok_idx,t in enumerate(insn) if v['segment_label'][insn_idx][tok_idx]==1 \
                                                   and not utils.is_special_token(t)], join=True))
        break

    insn_embeds = []
    for e_idx, embed in enumerate(embeds):
        insn_embeds.append( np.mean([e for _idx,e in enumerate(embed) if v['segment_label'][e_idx][_idx]==1], axis=0) )    

    down_res = TSNE(n_components=2).fit_transform(insn_embeds)
    utils.plot_2d_scatter(down_res, insn_strs, cl_args['plt_path'])

