# feedback-prize-ell
This is repository for tracking Feedback Prize - ELL kaggle competition

Public LB: 177th  
Private LB: 203rd (Bronzes Medal)

## What works
> - Differential learning rates
> - Layer-wise learning rate decay
> - Cosine scheduler with 25% warmup of the first epoch (about 5% of all steps)
> - Reinitialzing last layer
> - Multi-sample dropout
> - Mean pooling on longformer
> - Concatnated attention head with mean pooling on others architectures
> - AWP
> - Pseudo labelling
> - Seeds blending
> - Mixing different max_len ensemble models

## What did not work
> - Gradient clipping
> - Ranger21 
> - Madgrad and MirrorMadGrad
> - Freeze some layers
> - SWA
# Models
| Models                                   | CV       | Public LB | Private LB |
| ---------------------------------------- | -------- | --------- | ---------- |
| microsoft/deberta-v3-base (seed 42)      | 0.452696 | 0.44      |            |
| microsoft/deberta-v3-base (seed 12)      | 0.453501 | 0.44      |            |
| microsoft/deberta-v3-base (seed 0)       | 0.45297  | 0.44      |            |
| microsoft/deberta-v3-base (seed 42pl1)   | 0.451444 | -         |            |
| microsoft/deberta-v3-base (seed 12pl1)   | 0.452189 | -         |            |
| google/bigbird-roberta-base (seed 42)    | 0.460883 | 0.44      |            |
| google/bigbird-roberta-base (seed 12)    | 0.461839 | 0.44      |            |
| google/bigbird-roberta-base (seed 0)     | 0.461794 | -         |            |
| google/bigbird-roberta-base (seed 42pl1) | 0.460131 | -         |            |
| microsoft/deberta-v3-large (seed 42)     | 0.452084 | -         |            |
| microsoft/deberta-v3-large (seed 12)     | 0.453944 | -         |            |
| microsoft/deberta-v3-large (seed 0)      | 0.45297  | -         |            |
| microsoft/deberta-v3-large (seed 42pl1)  | 0.45039  | -         |            |
| microsoft/deberta-v3-large (seed 12pl1)  | 0.451133 | -         |            |
| roberta-large (seed 42)                  | 0.457107 | -         |            |
| roberta-large (seed 12)                  | 0.456173 | _         |            |
| roberta-large (seed 0)                   | 0.456481 | _         |            |
| roberta-large (seed 42pl1)               | 0.457107 | -         |            |
| roberta-large (seed 12pl1)               | 0.456173 | _         |            |
| allenai/longformer-large-4096 (seed 42)  | 0.45473  | _         |            |
| allenai/longformer-large-4096 (seed 12)  | 0.453414 | _         |            |

pl1: pseudo-labels 1 round

<!-- ![](misc/models_ensembling_diagram.jpeg) -->

# Notebook
These notebooks can run on google colab by using your kaggle api and google drive.
1. [fb_ell_trainb](notebook/fb_ell_trainnb.ipynb) &rarr; train model.
1. [fb_ell_mk_pseudo_label](notebook/fb_ell_trainnb_mk_pseudo_label.ipynb) &rarr; generate pseudo-labels in leak-free manner. (using subset of the first feedback competitions)
1. [fb_ell_pretrain_pseudo_label](notebook/fb_ell_trainnb_pretrain_pseudo_label.ipynb) &rarr; pretrain model on generated pseudo-labels.
1. [fb_all_finetune_pseudo_label](notebook/fb_ell_trainnb_finetune_pseudo_label.ipynb) &rarr; finetune the  pretrained models.
1. [fb_ell_cal_cv_nb](notebook/fb_ell_cal_cv_nb.ipynb) &rarr; generate out of fold prediction and calculate ensembled models score.
1. inference notebook &rarr; coming soon