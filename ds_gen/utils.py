from os import listdir
from os.path import isfile, join
import argparse
import sys
import csv

# read CL arguments
def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-rp", help="bin directory path", required=True, \
                        dest="bin_dir", metavar="") 
    parser.add_argument("-wp", help="bin directory path", required=True, \
                        dest="ds_write_path", metavar="") 
    #parser.add_argument("-sz", help="number of basic blocks written to each file", required=True, \
    #                    dest="bb_per_file", metavar="") 
    args = vars(parser.parse_args())
    return args

# generator over single-level directory containing binaries
def get_bin_path(dir_path: str) -> str:
    for f in listdir(dir_path):         
        yield join(dir_path, f)

# write one instruction per line, with empty row delimiter between basic blocks
def write_uncat_asm(ds_insns: list, write_path: str):
    with open(write_path, 'w') as fh:
        writer = csv.writer(fh, delimiter=',')
        for bb in ds_insns:
            for insn in bb:
                writer.writerow(insn)
            writer.writerow([])

# write two instructions per line, tab-seperated, in plain text format 
#   -last line will not be written, since insn pairs are required
def write_concat_asm(ds_insns: list, write_path: str):
    with open(write_path, 'w') as fh:
        for bb in ds_insns:
            bb_sz = len(bb)  
            
            if bb_sz < 2:
                raise RuntimeError('basic block with fewer than minimum # of insns encountered at write') 
            for insn_idx, insn in enumerate(bb):
                if (insn_idx == bb_sz-1):
                    break
                concat_insns = '\t'.join([' '.join(insn), ' '.join(bb[insn_idx+1])]) + '\n'
                fh.write(concat_insns)     
