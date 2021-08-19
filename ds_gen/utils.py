from os import listdir
from os.path import isfile, join
import argparse

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
