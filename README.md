# feedback-prize-ell
This is repository for tracking Feedback Prize - ELL kaggle competition
<!-- &#9745; for check --> 
<!-- &#9744; for uncheck -->
| Model | Weighted | CV | Public LB | Private LB |
| ----- | ----- | ----- | ----- | ----- | 
| microsoft/deberta-v3-base (seed 42) | - | 0.4526968 | 0.44 |  |
| microsoft/deberta-v3-base (seed 12) | - | 0.4535011 | 0.44 |  |
| microsoft/deberta-v3-base (seed 0)  | - | 0.4529697 | 0.44 |  |
| microsoft/deberta-v3-base (3 seeds) | N | 0.449868  | 0.43 |  |
| microsoft/deberta-v3-base (3 seeds) | Y | 0.449862 | - |  |
| google/bigbird-roberta-base (seed 42) | - | 0.4608834 | 0.44 |  |
| google/bigbird-roberta-base (seed 12) | - | 0.46183953 | 0.44 |  |
| google/bigbird-roberta-base (seed 0)  | - | 0.46179414 | - |  |
| debertav3b_3s + bigbird-base_s(42, 12) | N | 0.449401 | 0.43 |  |
| **debertav3b_3s + bigbird-base_s(42, 12)** | Y | 0.448970 | 0.43 |  |