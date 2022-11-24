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