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

## Assets
**resources.zip** contains:
* instruction datasets from several binutils (train) and diffutils (test) binaries
* x86-64 WordVocab pickles ('small': 17548, 'medium': 45852, 'large':64953)
* BERT (hidden: 128, layers=8, attn_heads=8) checkpoint file, pretrained on binutils instructions, using 'large' vocabulary
    * Trained on 3.2 million instruction pairs
    * Mini batch size of 768, with gradients accumulated over 64 mini batches per optimization step
    * Model trained for 24 epochs, 1584 optimization steps

## System Specs
Model was pretrained on AWS **g4dn.4xlarge** EC2 instance, and similar specs will be necessary for pretraining BERT with comparable dimensions
* Nvidia T4 GPU (16 gb)
* 64gb RAM

## Usage
1. Generate dataset from binaries
> python3 gen_ds.py -rp \<path to single-level directory of binaries\> -wp \<target path for dataset in flat file format\>
2. Generate WordVocab pickle
> python3 build_vocab.py -c \<corpus path\> -o \<output pickle\> -s \<vocab size\> -m \<min word frequency\> 
3. Pretrain BERT with language model target
> python3 train_bert.py -ds \<assembly ds path\> -vocab \<WordVocab pickle path\> -gpu \<0 or 1 to specify GPU\> -cpt \<model save dir\> -bat_sz \<batch sz\> -eps \<epochs\> 
4. Generate, instructions embeddings, map to 2d with T-SNE, and plot
> python3 gen_embeds.py -ds \<corpus path\> -vocab \<WordVocab pickle path\> -model \<model checkpoint path\> -bat_sz \<number of embeddings to gen & plot\> -plt \<plot path\>

## Example embeddings generated with supplied pretrained model
<p align="center">
  <img src = "https://i.imgur.com/ZcvLWB6.png">
</p> 
