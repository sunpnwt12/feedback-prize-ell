# What works
> - lr 1e-05
> - cosine scheduler with 5% warmup
> - epoch 5
> - batchsize 8
> - Layer-wise learning rate decay
> - reinitialzing last layer
> - multi-sample dropout
> - mean pooling
> - AWP from 2nd epoch
> - SWA from last 25% of the training
> 


# What did not work
> -  Gradient clipping
> -  Ranger21 
> -  Madgrad and MirrorMadGrad
> -  Freeze some layers
>  