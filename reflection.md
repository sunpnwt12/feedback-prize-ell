- Should have implement step validation 
    - or early stopping.

- Should spend more time on ensembling techniques
  - stacking using bayesian ridge

= Spending too much time on hyermeter tuning

- Should have tried differnt pooling strategies
  - LSTM pooling, CLS Token, etc.
  - MAany winners' solution mention about mixing differnt pooling in ensemble models.

- Spening more time understanding the architectures should help more.

- A lot of GPU time was wasted on bugs

- Pipeline was improved from the last competition but there are still some bugs
  - Configuration class should be sepearated.
  - dry_run and run_mode should be excluded from config file.
  - *self.config* should be enough for the whole trainer class.
  - getter function is a bit messy
  - gradient accumulation did not work as expect.

- Relied too much on pandas.

- *MultilabelStratifiedKFold* might cause some issue
  - This is still unknown. 
  - To prove this I will need to run everything again which is computation expensive.

- Should have tried train on concatnated pseudo-labels and given data.
  - only pretrain on pseudo-label and finetune on given data was applied

- AWP took so much time to train (almost twice as usual time)
  - should have tried turning it on at 2nd and later epoch.

- Models' max_len should not be so long, diversity should priotized over max_len

- Did not touch embeddding extracted SVR model.
  - this model should add more diversity to ensembled models.

# Works for some competitors

- Add special tokens.
  - or replace "\n" or "\n\n" with something else, like [NEWLINE], "|"

- Column-wise model

- decrease max_len per epoch

- Train on longer max_len and infer on shorter max_len
  - ex. train 2048 and infer 640
- Different loss rates per target
  - ex. {'cohesion':0.21, 'syntax':0.16, 'vocabulary':0.10, 'phraseology':0.16, 'grammar':0.21, 'conventions':0.16}