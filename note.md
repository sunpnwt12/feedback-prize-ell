# 9/15

- Starting with small model deberta-v3-xsmall.
  - Change its head from classification to regression task.

# 9/16

- Between the base model and the linear layer pooling is needed
  - In this case, MeanPool has been using since the past competitions.
- I need to understand more about the model(transformer) itself.

# 9/23

- Working on dataset/dataloader.
  - issue on putting data through the model.

# 9/26

- Padding size should be fixed to the max length of the full text (currently 512).
- Fixed issue with model taking data in (forgot to declare self as first arg).
- Next step is writing train loop.

# 9/27

- Finished simple Trainer class
- Next, setup lr_scheduler

# 9/29

- Arranged training print
- Need to fix loss calculating and lr scheduler

# 9/30

- Arranged print included metric (MCRMSE)
- Added txt logging and save model if metric gets better
- Fixed loss calculating
- Still need to fix lr scheduler ,and issue on max_len padding occurred

# 10/2

- Base trainer (xsmall model) runs through 5 fold without any problems
  - Log and best_score saving model also worked
  - Weird bugs on lr_scheduler

# 10/3

- Fixed bugs on lr_scheduler by switching to transfomer library's scheduler
- Finished basic training loop, start working on inference notebook

# 10/4

- Finished both basic train nb and inference nb.
  - OFM problem: need to implement gradient accumulation to fix this.

# 10/7

- Replaced newline \n with [BR] give a slightly worse result.
- Somehow reduce epoch to 5 give a better result (0.4565). Meanwhile, 10 epochs give around 0.46 and never goes below that.

# 10/11

- After multiple training sessions, the score is seemingling unstable with unknown reason.
  - Initial weights might cause an issue? because of its initial range is random.
  - Manaully initialize the weights (fc layer) seems to mild the oscillation, but it is still fluctuating. Maybe, it is caused by size of the data.
- Base model of the deberta seemed to work for a lot of people, sticking with the base model for a while.
- At this point, the training loss is showing some potential overfitting. Validation and LB score is almost useless because of small size of the data
- Currently, testing stability and reproducibility of the pipeline. (see if the result is reproducable, close reseult to some extent is acceptable)

  - There are slight different but acceptable.

- Effect of the loss on the training:

  > Note that PyTorch's L1SmoothLoss is both L1 and L2 loss. There is a hyperparameter beta=1 by default. This means when the model predicts within 1 of target, i.e. abs(target-pred)<beta=1, then the loss becomes L2 and when the model predicts greater than 1 of target, i.e. abs(target-pred)>=beta=1 then the loss becomes L1. So you can experiment changing beta. If we set beta=0 then we have purely L1 loss. And if we set beta=100 then we have basically L2 loss. See picture below. The straight lines are L1 and the curve is L2 and beta controls the size of curve.
  > ![Loss2](/misc/loss2.png)

- [TRY] change beta params in SmoothL1Loss to 0.025
  - [RESULT] it gave a worse result from 0.455xx to 0.457xx. Note that the loss acted differently from the usual beta=1 which normally 0.05~0.04 fro train, 0.10xx~0.11xx for valid to 0.16 for train and 0.35 for valid NEED FURTHER EXPERIMENT.
- [TRY] found that Adversarial Weight Perturbation (AWP) is working for some kagglers. Need to check out.

- [IDEA] remove escape character like \n.

- Found many people were using such a small lr, around 9e-6
  - [RESULT] It boosted CV from around 0.459xx to 0.455xx.
- also gradient checkpoint give a bit boost to the cv.
- need to look at max_len.

# 10/14

- Training is finally stable after:
    - add gradient_checkpoint.
    - decease lr to 9e-6.
    - exclude some weight from decaying.

- Tried Madgrad with two different lr (9e-6 and 5e-4) gave a worse result. Meanwhile, it is recommended for LM. It might need further experiment.

- AWP is somehow bothersome to implement. COMEBACK LATERFARTHER
- Tried Ranger21 with AdamW and Madgrad core.
    - significantly worse than plain AdamW and Madgrad.

- It is possible that tunning lr and max_len is the key to improve the cv.
- Have not yet exploring gradient clipping.
    - It was told that it helps stablize the training.

# 10/14

- Tweaking lr gave a slight improve to the fold0 (0.455099 -> 0.455003). Some columns were worse but better generalization in overall because of *phraselogy* and *grammar*  were improve. Note that this could be a lucky run. (95e-7)

- As expected put output_hidden_state to True make MeanPooling calculate correctly ,and it improve the fold0 (0.4550024, 0.454889 at the second run). 

- First try AWP:
    - worsen the fold0 (start at 2nd epoch).
    - Trying different hyperparams.

- Some weird bugs appeared when applied gradient clipping.
    - not training anything,loss do not decrease.

# 10/15

- Have not seen any improve to the fold0 experiment. (0.454889 vs 0.4550024)
- tried lr 2e-5, gave a worse result.
    - The training is quite sensitive to the lr.
    - In the sense that high lr (like 2e-5) would heavily overfit in the low layer of the model.
    - Other the other hand, based on the last successfully boosted lr, increase lr to 9.5e-6 from 9e-6 boosted the fold0's score.

- [Layerwise Learning Rate Decay](https://www.kaggle.com/code/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning/notebook)
    - Apply these tricks should help with the training stability and potentially overfitting problem.
    - After applied LLRD, it significantly boosted fold0 to 0.448972 and much better train and valid loss (0.0909xx and 0.1007xx respectively). That possibly means it does not overfit.
    - In this particular improved fold0, AdamW's eps was changed to 1e-6 (default was 1e-8). and smaller batch (2).

- Smaller batchsize boost the fold0 to 0.45025 which means it boosted around 0.004~0.005

# 10/17

- Freeze 2 first layers and re-initialize the last layer of the backbone gave 0.0001 boost to the fold0, and it help reducing training time by a margin (around 1 min per epoch included AWP).

- As expected, larger model boost the fold0's score by a lot (around 0.002) in exchange for much longer training time around 4 hrs for only 1 fold (included AWP) in this case.

- Decided not to use gradient clipping as it worsed the result.
- Seems like tuning hyperparams helps improve score according to the discussions in the competition forum.
    - Starts with the learning rate 9.5e-6 already have given a good result.
        - lr 3e-6 worsen the score so much (0.483xx)
        - **Sticking with 9.5e-6 for now.**
    - Next, max_len
        - padding max_len will normally increase the training considerably.
        - Currently, dynamic padding results the best score.
            - padding to the max length of the full text and truncating it just before feeding to the network.
        - max_len 512 is default value for the tokenizer, which does not require a lot of training time.
            - After a few training session, in this case, it can be seen that due to shorter length of the document, the model coud not correctly evaluate the longer document?
            - In the same pipeline that resulted 0.44883, max_len gave 0.46556.
            - Currently, sticking with the dynamic one.
    - batchsize
        - According to the random guys on the discussion forum, bigger batchsize needs higher lr than usual.

# 10/21
- Stashed experiment:
    - **FIRST** Tuning hyperparams
        - warm up only 5% of the whole training step, by change warmup_epoch to 0.25
        - increase lr to 1e-5 from 9.5e-6
        - increase batchsize to 4
        - reinit only last layer due to small dataset
        - [RESULT] substantially imprve the score to 0.445.

    - **SECOND** Using differnet head.
        - ConcatnatePooling last 4 layers
            - [RESULT] worsen, even more, the network is barely learning.
        - WeightedLayerPooling start from concatnate last 4 layers then from 4th to last layer
            - [RESULT] last 4 layers worsen, could possibly be improved if the hyerparams is properly tuned.
        
    - **THIRD**
        - linear_warmup -> It is seen to be used by all across winner's solution.
            - [RESULT] did not help improving the score (0.447x).
            - [OBSERVATION] The lr decreasing rate is not smooth as cosine one.
        
    - **FOUTH**
        - multi sample dropout head

    - **FIFTH**
        - gradient_clipping = 0.5
            [RESULT] somehow the network is not learning.

# 10/23

- weird thing happned: when shuffle the data in dataloader the score decreases.
    - from 0.455 to 0.448.
    - [OBSERVATION] As the train loss is almost similar to the valid loss. It is possible that not shuffling the data caused a overfitting.

# 10/25

- Increase batch size from 4 to 8
    - [RESULT] increase the score from 0.448 to 0.447
- Integrated SWA with swa_lr 1e-6 (smaller than than the overall lr) with linear annealing
    - [RESULT] greatly increate the score from 0.447 to 0.445. (train loss: 0.0973, valid loss: 0.0993)
        - It should lead to better generalization for the model.
- SWA with the same setting except anneal strategy, changed to cosine.
    - [RESULT] everything is almost the same. if anything it marginly improved by 0.00001. This might only be the case by the randomness. 

# 10/26
- Tried and succeeded in running dataparallel on T4x2. It it BLAZINGLY FAST. However, it became extremely slow when turn gradient checkpointing on.
    - Due to this, batchsize needs to be decreased but that hurt the performance of the model. It cant be fixed do more hyperparams tuning. I decided not to do that because it could take so much time go through tuning process again. 
    - Trying gradient accumulation bs4x2.
        - Did not work. Back to P100.

