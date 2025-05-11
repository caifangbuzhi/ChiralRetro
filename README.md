# ChiralRetro: Chirality-Sensitive Encoding and Enhanced Pathway Strategies in Retrosynthesis
The code was built based on [Graph2Edits](https://github.com/Jamson-Zhong/Graph2Edits). Thanks a lot for their code sharing!

This is the official [PyTorch](https://pytorch.org/) implementation for our model. Existing methods lack effective mechanisms to integrate chiral information and overlook the optimization of synthesis routes for multiple chiral center compounds. To address this challenge, we propose a novel molecular structure encoding algorithm to capture the chiral information of molecules.

## Setup

```
conda env create -f environment.yml
source activate chiral_retro
```
## Prepare Data
```
python preprocess.py --mode train
python preprocess.py --mode valid
python preprocess.py --mode test
python prepare_data.py --use_rxn_class # Known reaction type
python prepare_data.py                 # Unknown reaction type
```
## Train 
Known reaction type
```
python train.py --dataset uspto_chiral --use_rxn_class 
```
Unknown reaction type
```
python train.py --dataset uspto_chiral 
```
## Evaluate
Known reaction type
```
python eval.py --use_rxn_class --experiments exp --epochs epoch
```
Unknown reaction type
```
python eval.py --experiments exp --epochs epoch
```
The "exp" refers to the experiment name, such as "07-01-2025--21-51-12". The "epoch" refers to the experiment batch, such as "epoch_1.pt".
