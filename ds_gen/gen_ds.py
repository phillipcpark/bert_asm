import angr
import sys
import csv
from utils import parse_args, get_bin_path 

#
# this module creates a flat file formatted dataset of tokenized instructions
#   -Usage:
#      -rp <path to single-level directory containing binaries>
#      -wp <path to target destination for writing dataset>
#
#   -CSV output format:
#      <comma-separated instruction tokens>
#      <comma-separated instruction tokens>
#      |
#      |
#      <empty row block delimiter>
#
#   -Notes:
#      -all blocks for all binaries are concatenated into one file
#

# from single string representation of instruction, tokenize instruction and operands
def tokenize_insn(insn: str) -> list:
    # first token is address
    raw_tokens = insn.split('\t')[1:] 
    proc_tokens = []

    # skip insns 'nop' 'ret' for now   
    if (raw_tokens[-1] == ''):
        raise RuntimeError('instruction without operands encountered')   

    for raw_t in raw_tokens:
        raw_t = raw_t.split(',')                 

        # if 'ptr' token is found, need to concatenate with previous size operand
        found_ptr   = False        
        temp_tokens = []

        for child_t in raw_t:            
            child_t = child_t.split(" ")

            for t_idx, sub_t in enumerate(child_t):
                if (sub_t == ''):
                    continue
                if sub_t == 'ptr': 
                    # previous token specified ptr sz
                    temp_tokens[-1] += sub_t                   
                    temp_tokens.append(''.join(child_t[t_idx+1:]))  
                    found_ptr = True
                    break 
                else:
                    temp_tokens.append(sub_t)
            if (found_ptr):
                break              
        if found_ptr:
            proc_tokens += temp_tokens
        else:
            proc_tokens += [child_t.replace(" ", "") for child_t in raw_t]
    return proc_tokens

# generate block-delineated lists of instruction tokens, from CFG
def asm_from_cfg(cfg, proj) -> list:
    asm_blocks = [] 

    for nidx, node in enumerate(cfg.nodes()):
        block = proj.factory.block(node.addr)
        bl_cs = str(block.capstone).split('\n')

        block_insns = []
        for insn in bl_cs:
            try:
                block_insns.append(tokenize_insn(insn))
            except RuntimeError:
                continue
        if len(block_insns)>1:
            asm_blocks.append(block_insns)
    return asm_blocks 
             
#
# from binary directory path, deassmble all basic blocks in CFG and write as CSV
#
if __name__=='__main__':
    cl_args      = parse_args()
    bin_dir_path = cl_args['bin_dir']
    write_path   = cl_args['ds_write_path']
    
    ds_insns = []     
    for bin_path in get_bin_path(bin_dir_path): 
        print("\ndeassembling " + bin_path)
        proj      = angr.Project(bin_path, load_options={'auto_load_libs': False})
        cfg       = proj.analyses.CFGFast()
        deasm_bbs = asm_from_cfg(cfg, proj)
        ds_insns += deasm_bbs

    with open(write_path, 'w') as fh:
        writer = csv.writer(fh, delimiter=',')
        for bb in ds_insns:
            for insn in bb:
                writer.writerow(insn)
            writer.writerow([]) 

