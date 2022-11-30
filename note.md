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

- After multiple training sessions, the score is seemingly unstable with unknown reason.
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
  - [RESULT] it gave a worse result from 0.455xx to 0.457xx. Note that the loss acted differently from the usual beta=1 which normaly 0.05~0.04 fro train, 0.10xx~0.11xx for valid to 0.16 for train and 0.35 for valid NEED FURTHER EXPERIMENT.
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

- AWP is somehow bothersome to implement. COMEBACK LATER FOR FURTHER EXPERIMENT
- Tried Ranger21 with AdamW and Madgrad core.
    - significantly worse than plain AdamW and Madgrad.

- It is possible that tunning lr and max_len is the key to improve the cv.
- Have not yet exploring gradient clipping.
    - It was told that it helps stablize the training.

# 10/14

- Tweaking lr gave a slight improve to the fold0 (0.455099 -> 0.455003). Some columns were worse but better generalization in overall because of *phraselogy* and *grammar*  were improve. Note that this could be a lucky run. (95e-6)

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
            - later found out that the training dataset did not shuffle

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
- Tried multi-sample dropout p=0.3
    - slighly worsen. Determined by the result of the first epoch. FURTHER EXPERIMENT ex. p=0.2

- Tried different learning rate in backbone layers and the lower layers
    - bb 1-e5, ll 2-e5
    - [RESUTL] Gained 0.0002 from the previous one from 0.44588 to 0.44564

# 10/27
- Different max_len for different fold
    - [RESULT] max_len 512 for fold 2 -> did not improve

- [HOLDING] Pseudo labelling from the previous competition.
    - quite easy to implemention but require a lot of computing power.
    - Need to be careful when choosing the data in integrate into the original data.
        - Ex. the distribution.

# 10/28
- Change number of fold to 4 (Currently 5)
- Pseudo Labelling from the previous (pipeline almost finish, make the 1st round train remaining)

# 10/29
- 1.AttentionHead and mean pooling concat on the fold 2 (version 119 as a comparision sample)
    - [RESULT] 
        - max_len 512: improved the score on the fold 2 from 0.4605 to 0.4573
        - max_len 1024: the score slightly dropped from 0.4573 to 0.458343, seems like max_len must depend on fold
            - ex. fold 0 have achieved the low error (0.4456) with dynamic padding (max_len 1428). Seemingly, it does not fit with the other folds, like fold 2. So the best possible way to achieve higher cv is to pick the right max_len the pariticular fold.
                - To prove this theory, max_len 512 on fold 0 will be the next experiment.
                - run cv with attentionhead concatnated to mean pooling, different max_len for different fold [1428, 512, 512, 1428, 1428]
                    - fold 0 got worse 0.452

- Only swapping where does the layer should be reinitialized impact a lots on the score
    - reinit after declare poolings (0.450x)
        - [RESULT] After trained on every folds, even placed reinitization method at the specific step that allow fold 0 to achieve 0.4456. the remaining folds got so much worse.
            - ((1, 0.4595), (2, 0.4582), (3, 0.4572), (4, 0.4555)). This might mean that the current setup overfitting fold 0.

# 10/30
- Turn off AWP in fold 2 (0.460581 at 3nd epoch)
- Turn off AWP and SWA on fold 2 (0.460586 at 3nd epoch)

- Eveyrything is so confusing. Restarting things. (Everhing below this run with max_len 512)
    - start again with max_len 512 on fold 2 for speed and SWA because it does not require big overhead calculation. (around 32 mins per fold)
        1. start with 0.4605.
        2. changed multi-sample dropout to range of 0.1 to 0.5. 
            - increase to 0.4600
        Note - Considering changing test subject to fold 4 as it is likely represent the mean of the splits. In other words, it always gives the closet score to the mean of the score.
        3. switched to fold 4
            - with 0.4541
        4. changed mean_pooling to concat_attn_mean_pooling
            - improved to 0.4538
        4. chnaged concat_attn_mean_pooling to concat_all_attn_mean_pooling
            - deceased to 0.4545
            - stick with concat_attn_mean_pooling
        5. concat_wlp_mean_pooling
            - 0.4556
    
    - concat_attn_mean_pooling, backbone_lr 2-e5, lowlayer 3-e5
        - 0.4528
    - weights initialization (xavier_uniform, orthogonal)
        - xavier_uniform: 0.4476 -> An agressive choice
            - lb: 0.49
            - Maybe this is caused by either concatnating with attention head or xavier_uniform itself
            - Experimenting by change the pooling strategy back to mean_pooling: 0.452
        - orthogonal: 0.452047 -> A conservatiive choice
            - lb: 0.50
            - Likewise, if any thing, there is a change that xavier_uniform better.
        - Obviously, the score depends on how the weights were initialized before the training.
    
    - **NEED TO FIND BETTER WAY TO MAINTAIN CONFIG BETWEEN TRAINING AND INFERENCING NOTEBOOK**
        - write export fundtion: 
            - make inference notebook load config from the yaml file.
            - export to yaml file, Then, loading it using AttrDict, which can access the members (attributes) by dot.
        - ~~write showing cv table in the inference notebook.~~

# 10/31
- add LayerNorm (Summission is taking so much time, more than 2 hours on cpu) <- missed configuration batch_num instead of 1.

- fold 0: 0.4501, LB: 0.46
    - Have seen many people experience the same thing that even the local score improve the LB does not reflect that.
    - One odd thing that happened when add LayerNorm to the network is train_loss always higher than the valid_loss
    - Although, it is likely to improved the score from 0.452x to 0.450x

# 11/02
- Turn on the AWP
    - 5 fold [0.4493, 0.4555, 0.4571, 0.4509, 0.4486], CV:0.4524, LB: 0.44
    - train_loss around 0.131, valid_loss around 0.101

- Different between having a LayerNorm in the network and not
    - fold_0_score + 0.0002, LB lower than not having one.
    - Local score is more trustworthy. However, It should be aware that fold 0 is always achieving the best score out of the other fold.

- Followed [this](https://github.com/Danielhuxc/CLRP-solution) implemention of weighted layer pooling and attention head 
    - with max_len 1024, layernorm score: 0.4583

- Time to switch to 4 fold splitting
    - train without awp for speed, 

# 11/03
- exp 10 multidrop all ~~0.5~~ change to 0.3, 0.5 might be a bit too agressive.
    - reinit with the xavier normal
    - when reinitailize last layer of the back use normal initial weights
    - using kaiming for fc layer (turn off awp for speed)

- exp 11 orthoganol initialization.: fold0 0.4522

- exp 12 bb_lr = 1e-5  ll_lr = 2e-5: fold0 0.4542

- exp 13 normal dists initialization: fold0 0.4512

- exp 14 same setup but turn on AWP: fold0 0.44930413365364075 

# 11/04
- exp 15 turn swa on lr=1e-5: fold0 0.45103

- exp 16 add layernorm: fold0 0.4499546

- exp 17 AttentionPooling with GELU instead of tanh concat with MeanPooling: fold0 0.4514607

- exp 18 AttentionHeaad with original tanh concat with MeanPooling: fold0: fold0 0.45265332

- exp 19 AttentionHeaad with original tanh concat with MeanPooling (ininitilize with normal dists): fold0 0.45265332

- exp 20 increase epoch to 6 and start swa from the 3rd epoch: fold0 0.45049605

- exp 21 try AttentionPooling using last hidden state: fold0: 0.44966474
    - Showing good potential.
    - concat them?

# 11/06
- exp 22 WeightedAverage of all encoders outputs with newly implemented WeightedLayerPooling referring from [this](https://github.com/Danielhuxc/CLRP-solution/blob/main/components/model.py) and [this](https://github.com/oleg-yaroshevskiy/quest_qa_labeling/blob/master/step5_model3_roberta_code/model.py). After that put it through new attention pooling that is using gelu.
    - not learning anything. (loss is not decreasing)
    
- exp 22.1 fixed where attn pooling should be calculated before dropout.

- exp 22.2 weights_init.data[:-1] = -3

- exp 22.3 double current lr to 4e-5 and 6e-5

- exp 23 change pooling strategy to concat_attn_mean_pooling (this time use new one with gelu): fold0 0.44938600063323975

- exp 24 turn swa on with swa_lr 5e-6: fold0 0.45361838

- exp 25 multidrop_p to 0.5 from 0.3: fold0 0.44981873

- exp 26 put lr back to 1e-5 and 2e-5 and turn on SWA(trying to reproduce past result)

# 11/08
- run all 4 fold with exp23 setup
    - [0.44938636, 0.45520418, 0.45231235, 0.44644296]
    - CV: 0.45288518 

- exp 27 same setup as exp23 except 4 epoch and start AWP from the start:fold0 0.4480045
[IDEA] averaging model's last few checkpoint.
    - can do it manually or using the provided one.

- exp 28 start AWP at the 2nd epoch: fold0 0.4486684

- exp 29 same setup as exp27 but start SWA from step 1300/1464 (last 0.112): fold0 0.4491706 lb: 0.44 (higher than exp27, higher than CV_MODE exp23)

- exp 30 using manual SWA by average 3 different checkpoints [1200, 1300, 1400].

- exp 31 same setup as exp27 but run on fold 2 0.4598968

- Candidate for ensembling model
    - BigBird
    - Longformer
    - Funnel Transformer

- exp 32 same as exp31 multi-sample dropout from all 0.3 to range of 0.1 to 0.5: fold2 0.45964053
    - stick with this


# 11/09-11
- Lost all the tracking between the dates (note is inside WSL and it brokes)
    

# 11/12
- exp 43s12
    - 4 fold [0.46427143, 0.45597076, 0.4439039, 0.4494147] CV:0.45350114 LB: higer than seed 42
    - when emsemble this with seed 42, the lb score increase and became the first among all summits.

- exp 43s0
    - 4 fold [0.45604673, 0.45623007, 0.45081356, 0.44854555] CV: 0.4529697 LB: higher than seed 42 but lower than seed 12

- when blend all 3 seeds
    - CV: 0.449868, LB: 0.43 (Finally, but stil close to 0.440)
    - using optuna to tune weight for averaging CV: 0.449862S

# 11/13 
- exp 45 tried bigbird-roberta-base, seed 12: fold 3 0.46028018

- checking different between seeds(42, 12, 0)
    - CV chnages considerably by only changing seed.
    - So technically, included extra some features like unique_words altetinate how the data is spliting. which means alternated splited data might be equivalent to some random seed. Can not comfirm this because haven't tried yet.

- exp 46 tried all allenai/longformer-base-4096, seed 12: fold 3 
    - Maybe longformer has different architecture, error occurred after 120 steps.

- exp 47 back to bigbird-roberta-large seeed 12: fold 0 0.47063088
    - large model start to overfit very quick

- Found bugs in gradient accumulation: FIXED
  - Regular Nb (debertav3)
  - Bigbird Nb
  - FunnelTransofmer Nb
  - colab/regular Nb
  - colab/bigbird Nb

- exp 48 based on exp45, increase learning rate to 3e-5, 5-e5: fold 3 0.45909524

- exp 49s12 run cv based on exp48, to testify how it would behavior.
    - 4 fold [0.4735262, 0.4650949, 0.44908154, 0.45909524] CV: 0.46183953 LB: 0.44 quite below.
    - the cv itself is pretty bad but when ensemble to the first 3 deberta model cv increase from 0.449862(weighted) to 0.449352(weighted) (by 0.0006)

# 11/14

- Seems like ensemble deberta and bigbird (both are base models) is working together very well
  - correlation between cv and lb is good (if cv increases, lb is also increase)

- exp49s42
  - 4 fold [0.4536219, 0.46550596, 0.4655917, 0.45861617] CV: 0.4608834 LB: 0.44 higher than seed 12 but still lower than deberta-v3-base model.

- As expected, As cv increase, the lb also increase.

# 11/15
- Somehow selecting only high score folds in both bigbird and debertav3 model. increase lb score.
  - This is very dangerous because it is likely to overfit the pulibc test dataset.

- exp 50 trying deberta-v3-large seed 42, bs 4 ga 2 ,max_len 1024 : fold 0 0.44656098
    - Don't know why but the score is looking quite good on lb 0.44 higher than exp43s0
    - complete the cv would be a good idea.

# 11/16
- [TODO] complete exp50 4 fold [1, 2, 3] left.
    - [DONE] [1, 2, 3]

- next model would be longerformer or deberta-v2-base.
- deberta-v2-base does not exist so deberta-base is only choice for base model.

- exp 51 trying deberta-base seed12 bs 5 beause deberta-base has more paameters (100 vs 86)
    - Cannot find a way to fixed OOM error. Decided to skip for now.

- After submitted exp43+exp49s(42, 12)
- Found bugs in accumulatte_step configuaration
  - if accumulate_step = 0, gradient accumulation is not used
  - Howver, if accumulate_step == 1 only loss / accumulate_step will be triggered but not optimizer.step()

# 11/17
- included deberta-v3-large model is significantly increase cv to 0.4479 for simple averaging and 0.4471 for weighted averaging.
    - Aligning CV and LB is quite a relief.
    - that being said, it is likely that selecting particular fold in the full 4 folds train give a better result
      - in this case, with exp43 + exp49s42,12 + exp50s42f0 has higher lb rank
        - this weighted submit jump to sliver medal area.
      - again, this is very dangerous becuase the very selected folds are more likely to overfit to public testset

- [CONSIDERING DIRECTIONS]
  - train one more large model?
  - train differnt architecture like bart or longformer or roberta?
  - train deberta-base (v1)?

# 11/18

- exp 50s12 train deberta-v3-large model on seed 12
    - 4 fold [0.44656095, 0.4565786, 0.45514414, 0.4498687] CV: 0.4498687
    - after submit with weighted averaged of exp43+exp49s42,12+exp50s42,12(CV: 0.446733), it is clear that it improved when included seed 12 of the large model. What strange is simple averaged one is ranked higher than the weighted one, maybe, because it is a bit overfitting.
    - what beyond this submit is likely to be overfitting submit. ex. selecting particular fold in the full 4-5 folds.

- oor0 (Out of record)
    - previously trained 5 fold 5 epoch swa last 0.25 of the training
    - mean_pooling, multiple sample dropout 0.1
    - weighted ensemble increase cv to 0.44707 from 0.44713, but it does not reflect on lb

- exp 52 tried roberta-base seed 0: fold 0 0.459063650
    - due to shorter max_len 512, per fold is very fast around 32 mins per fold
    - further experiment, 3 seeds of this model

- It might be better to mix different acrchitectures in final submit
  - since roberta is in bert family the result might not be significantly improved or in worse case scenario, it overfits because the way bert family still have similar parts inside themself, which means, it could train toward in similar ways even if they are different in detail.

- exp 53 tried longformer-base, seed 42 bs8 max_len 1024
    - so the errors or what ever caused an error to the previous exp is max_len. (last time was 1426, it is not 512*x)

# 11/19

- exp 54s42 roberta-large
    - 4 fold [0.45199814, 0.45807615, 0.46153107, 0.45669726] CV: 0.45876816
    - included this to the ensembled model increase score to 0.44657 (simple) and 0.44601 (weighted)
    - in public lb mean still higher than weighted one (select both of them in final subs) and last final sub is fold-selected one
    - add one more model should help boost the score

- exp 55 funnel-transformer/large
    - the training loss stuck around 0.3xxx and it did not decrease afterward.
    - gave up on this

# 11/20

- exp 56 bigbird-roberta-large
    - didn't finish as first fold did not show good sign

- exp 57 longformer-large mean pooling
    - CV 0.45473

# 11/21

- exp 57c longformer-large concat_attn_mean_pooling: fold 0 
    - didn't finish as first fold did not show good sign

# 11/23

- exp 58 pretrain deberta-v3-base max_len 768 on generated **pseudo-label** seed 42
    - pseudo-labels 1st round (exp58s42 and exp58s12) deberta-v3-base
    - chose max_len 768 because of puporse of diversity
    - seed 42
        - 1st round
            - tried with only 1 epoch and finetune on given dataset 3 epoch
                - fold 0: 0.44800234 --> 0.44696045	
                - fold 1: 0.4555346  --> 0.45513305
                - fold 2: 0.45989653 --> 0.45806667	
                - fold 3: 0.44715628 --> 0.44541147
                - CV:  0.452647447 --> 0.451443940 (by 0.0012)

    - seed 12
        - 1 epoch pretrined and 3 epoch finetune
            - fold 0: 0.46427143 --> 0.463923
            - fold 1: 0.45597076 --> 0.454694
            - fold 2: 0.4439039  --> 0.4426024
            - fold 3: 0.4494147  --> 0.4470898
            - CV: 0.453390 --> 0.4521891 (by 0.0012009)

# 11/24
- [TODO]
  1. ~~make oof df from the 1st round (can be done in colab? load model from google drive)~~
  1. ~~decide what to include in 2nd round~~
  1. ~~Move pseudo-labels flow into colab~~ (mk, ~~pret, finet~~)  (cannot move mk because all trained model is uploaded to the kaggle dataset, unless download all models to colab)

- exp 59
    - pseudo-labels 2nd round (exp58s42 and exp58s12)

- What if bigbird didn't go well with the concatnated heads?
- exp 60 bigbird-roberta-base seed 42 with **mean pooling** increase lr to 3e-5 and 4e-5
    - comparing with concat_attn_mean_pooling
    - fold 0: 

- exp 61 pretrain bigbird-roberta-base max_len 768 on generated **pseudo-label** seed 42
    - pseudo-labels 1st round (exp58s42 and exp58s12) for bigbird-roberta-base
    - seed 42
        - tried with only 1 epoch and finetune on given dataset 4 epoch (bigbird needs longer and more agressive train)
            - fold 0: 0.4536219  --> 0.4523439
            - fold 1: 0.46550596 --> 0.46609166
            - fold 2: 0.4655917  --> 0.46358514
            - fold 3: 0.45861617 --> 0.4582764
            - CV: 0.4608834 --> 0.46013138 (by 0.00075)
    
- exp 62 pretrain deberta-v3-large max_len 768 on generated **pseudo-label** seed 42
    - seed 42
        - fold 0: 0.44656095 --> 0.44457
        - fold 1: 0.45514414 --> 0.45244
        - fold 2: 0.4565786 --> 0.45539
        - fold 3: 0.4498687 -->  0.44897
        - CV: 0.452084 --> 0.4502 (by 0.001884)
    - seed 12
        - fold 0: 0.46305826 --> 0.45956
        - fold 1: 0.45678416 --> 0.45486
        - fold 2: 0.44530618 --> 0.44161
        - fold 3: 0.45020148 --> 0.44804
        - CV: 0.453944 --> 0.451133 (by 0.002811)

- cv and lb are starting misaligned from where roberta-large seed 12 was included
    - public lb datset is quite small. there is a good chance this particular dataset doesn't represent the whole dataset (private lb dataset).
    - so trust cv is a way to go.

# 11/25
- [TODO]
    1. ~~finetune deberta-v3-large (exp62) using pretrained models in google drive~~
    1. ~~pseudo-label roberta-large (as it is so fast to train)~~

- exp 63 pretrain roberta-large max_len 512 on generated **pseudo-label** seed 42
    - seed 42
        - fold 0: 0.45199814 --> 0.45017728
        - fold 1: 0.45807615 --> 0.45647225
        - fold 2: 0.46153107 --> 0.45822012
        - fold 3: 0.45308042 --> 0.45669726
        - CV: 0.457107 --> 0.454488 (by 0.002619)
    - seed 12
        - fold 0: 0.4629476  --> 0.4623566
        - fold 1: 0.46175924 --> 0.4607107
        - fold 2: 0.44510302 --> 0.4439592
        - fold 3: 0.45451307 --> 0.4520159
        - cv: 0.45617303 --> 0.45485672 (by 0.00131631)

# 11/26
- [TODO]
    1. ~~calculated cv included with deberta-v3-large pl1~~
    1. ~~make pretrained labels seed 12 and make it datasets (on kaggle kernels)~~
    1. ~~pseudo-labels roberta-large seed 12 (on colab)~~
    1. ~~pseudo-labels deberta-v3-base seed 12 (on kaggle kernels)~~
    1. if have time left pseudo-labels longformer-large seed 12? but what max_len
    1. if have time left train roberta-large seed 0?

- How about deberta-v3-large seed 0? (with fixed one)

- retrain all seed 0 with fix one?
    - as it is not included in any pseudo-labelling process
    - or instead use this new fix train pl 2nd round
      - will it leak?
        - it might not leak because it introduce new pl=data to the fresh downloaded models
        - It might leak because those new pl-data are made from the model that knows about these data


# 11/28

- exp 65 deberta-v3-small max_len 768 seed 42
    - 4 fold [0.45532247, 0.46335778, 0.462499, 0.4542868] CV: 0.45886650681495667
    - CV is not bad comparing to the other architectures.
    - It did not contribute to ensembled models, perhaps, because it did not introduce any diversities.

- after trained and tried many models, this should be final submission models (**selected**)
    - it comprised of 14 models, 5 different architectures.
        - **deberta-v3-base 3 seeds (42, 12, 0)**
            - 42 and 12 are pseudo-labels 1 round trained models
        - **deberta-v3-large 3 seeds (42, 12, 0)**
            - this seed 0 is a little bit unique from others, since i accidently spilted thte data with one-hot encoded (the mentioned fix one)
        - **deberta-v3-large 2 seeds (42, 12)**
            - these models are pseudo-labels 1 round trained models
            - mixed them with the original one as it imporved the score
        - **bigbird-roberta-base seed 42**
            - trained 3 seeds but others did not improve cv
        - **roberta-large 3 seeds (42, 12, 0)**
            - seed 42 is pseudo-labels, trained 12 but did not improve cv
        - **longformer-large 2 seeds (42, 12)**
    - ensembled all models with simple mean cv: ***0.445133***
    - weighed ensembled cv: ***0.4450627*** 
    - (11/30) 2nd best submission
    - public lb: 0.437602
    - private lb: 0.436404

# 11/29

- Last day before deadline. (30 November 2022 11:59 UTC)

- Try to finish exp 64s42 2nd round (CU left on colab)
    - deberta-v3-base seed 42
    - pretrained on different subset of fb1 data 
    - Even though it get a decent score, it does not improve cv when ensembled   

- Last submission (**SELECTED**)
    - included 21 models, 5 different architectures.
        - **deberta-v3-base 3 seeds (42, 12, 0)**
        - **deberta-v3-base pseudo-labels 2 seeds (42, 12)**
        - **deberta-v3-large 3 seeds (42, 12, 0)**
        - **deberta-v3-large pseudo-labels 2 seeds  (42 ,12)**
        - **bigbird-roberta-base 3 seeds (42, 12, 0)**
        - **bigbird-roberta-base pseudo-labels 1 seed (42)**
        - **roberta-large 3 seeds (42, 12, 0)**
        - **roberta-large pseudo-labels 2 seeds (42, 12)**
        - **longformer-large-4096 2 seeds (42, 12)**

  - Weighted ensembled CV: **0.4449692**
  - (11/30) 3rd best submission
  - Public LB: 0.437734
  - Private LB: 0.436439

# 11/30

- Competition ended.

- Public LB Rank is 178th and Private LB Rank is 203rd.

- Best submission (Did not select)
    - Included 10 models, 4 different architectures.
        - **deberta-v3-base 2 seeds (12, 0)**
        - **deberta-v3-base pseudo-labels seed 42**
        - **bigbird-roberta-base pseudo-labels seed 42**
        - **deberta-v3-large 2 seeds (42, 12)**
        - **deberta-v3-large pseudo-labels seed 42**
        - **roberta-large 2 seeds (42, 12)**
        - **longformer-large seed 12**

    - Weighted ensembled CV: **0.4454052**
    - Public LB: 0.437664
    - Private LB: 0.436382