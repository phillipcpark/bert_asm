# BERT x86-64 instruction embeddings
Note: all model files in arch subdirectory taken from https://github.com/codertimo/BERT-pytorch

## Dependencies
* Python3 
* pytorch
* CUDA (tested on 11.0)
* angr
* tqdm
* plotly
* kaleido

## Usage
1. Generate dataset from binaries
> python3 gen_ds.py -rp <path to single-level directory of binaries> -wp <target path for dataset in flat file format>
2. Generate WordVocab pickle
> python3 build_vocab.py -c \<corpus path\> -o \<output pickle\> -s \<vocab size\> -m \<min word frequency\> 
3. Pretrain BERT with language model target
> python3 train_bert.py -ds \<assembly ds path\> -vocab \<WordVocab pickle path\> -gpu \<0 or 1 to specify GPU\> -cpt \<model save dir\> -bat_sz \<batch sz\> -eps \<epochs\> 
4. Generate, asm embeddings, map with T-SNE, and plot
> python3 gen_embeds.py -ds \<corpus path\> -vocab \<WordVocab pickle path\> -model \<model checkpoint path\> -bat_sz \<number of embeddings to gen & plot\> -plt \<plot path\>
