# feedback-prize-ell
This is repository for tracking Feedback Prize - ELL kaggle competition
<!-- &#9745; for check --> 
<!-- &#9744; for uncheck -->
| Model                                                           | Weighted | CV       | Public LB | Rank LB |
| --------------------------------------------------------------- | -------- | -------- | --------- | ------- |
| microsoft/deberta-v3-base (seed 42)                             | -        | 0.452696 | 0.44      | 34      |
| microsoft/deberta-v3-base (seed 12)                             | -        | 0.453501 | 0.44      | 27      |
| microsoft/deberta-v3-base (seed 0)                              | -        | 0.452969 | 0.44      | 29      |
| microsoft/deberta-v3-base (3 seeds)                             | N        | 0.449868 | 0.43      | 24      |
| microsoft/deberta-v3-base (3 seeds)                             | Y        | 0.449862 | -         | -       |
| google/bigbird-roberta-base (seed 42)                           | -        | 0.460883 | 0.44      |         |
| google/bigbird-roberta-base (seed 12)                           | -        | 0.461839 | 0.44      |         |
| google/bigbird-roberta-base (seed 0)                            | -        | 0.461794 | -         |         |
| debertav3b_3s + bigbird-base_3s                                 | N        | 0.449995 | 0.43      |         |
| debertav3b_3s + bigbird-base_3s                                 | Y        | 0.448971 | 0.43      |         |
| debertav3b_3s + bigbird-base_s(42, 12)                          | N        | 0.449401 | 0.43      | 23      |
| debertav3b_3s + bigbird-base_s(42, 12)                          | Y        | 0.448970 | 0.43      | 20      |
| microsoft/deberta-v3-large (seed 42)                            | -        | 0.452084 | -         | -       |
| microsoft/deberta-v3-large (seed 12)                            | -        | 0.453944 | -         | -       |
| debertav3b_3s + bb-base_s(42, 12) + debertav3l_s42              | N        | 0.447958 | 0.43      | 18      |
| debertav3b_3s + bb-base_s(42, 12) + debertav3l_s42              | Y        | 0.447135 | 0.43      | 16      |
| debertav3b_3s + bb-base_s(42, 12) + debertav3l_s(42, 12)        | N        | 0.447214 | 0.43      | 13      |
| debertav3b_3s + bb-base_s(42, 12) + debertav3l_s(42, 12)        | Y        | 0.446733 | 0.43      | 14      |
| roberta-large (seed 42)                                         | -        | 0.457107 | -         | -       |
| roberta-large (seed 12)                                         | -        | 0.456173 | _         | -       |
| d_v3b_3s + bb-b_s(42, 12) + d_v3l_s(42, 12) + roberta_large_s42 | N        | 0.446579 | 0.43      | 6       |
| d_v3b_3s + bb-b_s(42, 12) + d_v3l_s(42, 12) + roberta_large_s42 | Y        | 0.446012 | 0.43      | 9       |
| d_v3b_3s + bb-b_s(42, 12) + d_v3l_s(42, 12) + rl_s(42, 12)      | N        | 0.446286 | 0.43      | 7       |
| d_v3b_3s + bb-b_s(42, 12) + d_v3l_s(42, 12) + rl_s(42, 12)      | Y        | 0.445787 | 0.43      | 11      |
| d_v3b_3s + bb-b_s42 + d_v3l_s(42, 12) + rl_s(42, 12)            | N        | 0.445981 | 0.43      | 8       |
| d_v3b_3s + bb-b_s42 + d_v3l_s(42, 12) + rl_s(42, 12)            | Y        | 0.445788 | -         | -       |