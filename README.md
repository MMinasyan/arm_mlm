# Pretraining Character Level Transformer for Armenian

Transformers are known for best performance models for NLP tasks today, and there are many pretrained models that can be used for various tasks.

However, when trying to build such models for Armenian language, one can encounter a problem that the model architectures working great on English won't work the same way on Armenian.
The main reasons are large vocabulary and extremely complicatated grammar.

For example, if a verb can form 3 or 4 tokens through morphological transformation in English, in case of an Armenian verb, the number of such tokens can be up to 300.
One possible solution for such problem is character level model.

# Methods and Techniques

python 3.9 and Tensorflow 2.10 are primary tools used for the project.

## Model

The model uses characters as input including Armenian and Latin letters, digits and other symbols in the token set.

The main body has the same architecture with self-attention models as BERT, with hyperparameters:

character embedding = hidden dimension = 256<br />
number of attention heads = 8<br />
number of attention layers = 8<br />
feedforward dimension = 1024


Number of trainable parameters in the model is 6,413,000 

## Data

Paragraphs from 22 different media sources with maximum lenght of 512 characters.<br />
training samples: 1,159,000<br />
validation samples: 145,000<br />
testing semples: 145,000

## Pretraining

The teqnique used for pretraining of the language model is random masking (15% of symbols):

Categorical Cross Entropy is applied as loss function, computated on masked characters.

Learning rate is bigger in the first epoch and dectreases during training process which includes 65 epoch. Batch size is 256.

## Results

Results show ~ 94.6 % accuracy rate over test samples.<br />
However, the model performance should be tested after finetuning for specific tasks

The work on this direction is in progress at the moment, and hopefully would be presented soon.

Pleese see the more details in arm_mlm.ipynb notebook.

