## 9/15

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

- base trainer (xsmall model) runs through 5 fold without any problems
  - log and best_score saving model also worked
