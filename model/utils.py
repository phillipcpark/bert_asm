from collections import Counter, OrderedDict
import argparse
import pickle
import csv
import sys

from plotly import graph_objects as go
import torch as th
from vocab import WordVocab 

#
def parse_args(for_train=True) -> dict:
    parser = argparse.ArgumentParser()

    parser.add_argument("-ds", help="path to dataset of concat insns", required=True, \
                        dest="ds_path") 
    parser.add_argument("-vocab", help="path to pickled vocab", required=True, \
                        dest="vocab_path")
    parser.add_argument("-gpu", help="1: use gpu, 0: cpu", required=True, \
                        dest="gpu", type=int) 
    parser.add_argument("-cpt", help="model checkpoint save dir path", required=True if for_train else False, \
                        dest="cpt_dir") 
    parser.add_argument("-model", help="model checkpoint load path", required=True if not(for_train) else False, \
                        dest="model_path") 
    parser.add_argument("-bat_sz", help="batch size", required=True, \
                        dest="bat_sz", type=int) 
    parser.add_argument("-eps", help="number of epochs", required=True if for_train else False, \
                        dest="epochs", type=int)
    parser.add_argument("-plt", help="plot path", required=True if not(for_train) else False, \
                        dest="plt_path")

    ##########
    # defaults
    ##########

    # determined using utils.get_max_seq_len
    MAX_INSN_LEN = 4
    SEQ_LEN      = 2*MAX_INSN_LEN + 3 
    
    args = vars(parser.parse_args())
    args.update({'seq_len': SEQ_LEN})

    return args

# load pickled WordVocab instance into memory 
def load_vocab(path: str) -> WordVocab:
    with open(path, "rb") as fh:
        return pickle.load(fh)

# used if manual Dataset in-memory instantiation desired
def load_ds(path: str):
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

# from list of basic block delineated token lists, get frequencies of all unique tokens
def sorted_tok_freqs(bbs: list) -> dict:
    all_toks = [tok for bb in ds for insn in bb for tok in insn] 
    tok_counts = Counter(all_toks)

    tok_counts = sorted(tok_counts.items(), key=lambda x: x[1], reverse=True)
    tok_counts = OrderedDict(tok_counts)
    return tok_counts 

# first 5 tokens are special tokens
def is_special_token(token):
    return token <= 4 

# in-place replacement of masked tokens (as returned by DataLoader) with their unmasked values
def replace_masked_tokens(masked_insns: th.Tensor, lm_labels: th.Tensor, mask_idx=4):
    for insn, mask_labels in zip(masked_insns, lm_labels): 
        for t_idx,token in enumerate(insn):
            if token==mask_idx:
                insn[t_idx] = mask_labels[t_idx]

# determine largest seq len from nested list of basic block delineated token sequences
def get_max_seq_len(insns_path: str) -> int:
    bb_insns = load_ds(insns_path)
    max_len  = 0

    for bb in bb_insns:
        for insn in bb:
            # pairs of insns are concatenated, delimited by tab
            insn = insn[0].split('\t')
            max_len = max(max_len, len(insn[0].split(' ')), len(insn[1].split(' ')))
    return max_len            

#
def plot_2d_scatter(vals: list, annots: list, write_path: str):
    figure = go.Figure()
    scatter = go.Scatter(x = [e[0] for e in vals], y = [e[1] for e in vals], \
                         text = annots, \
                         mode = 'text+markers', \
                         textposition="bottom center")

    figure.add_trace(scatter)

    figure.update_layout(title = { 'text': 'x86_64 BERT embeddings', 'x': 0.5, \
                                   'font': {'size': 32} },
                         width = 1600, \
                         height = 1200)

    figure.update_xaxes(title = { 'text': 'T-SNE dim 0', 'font': {'size':24} })
    figure.update_yaxes(title = { 'text': 'T-SNE dim 1', 'font': {'size':24} })
    figure.write_image(write_path)
                                
