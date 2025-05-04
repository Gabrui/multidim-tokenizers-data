# Multidimensional Tokenizers Training and Dataset generation

`Python 3.10.12`

## Training Tokenizers

The tokenizers are already trained on the `tokenizers` folder. But if the need arises to retrain, run the Jupyter Notebook [multi_tokenizer.ipynb](./multi_tokenizer.ipynb) .
There are other libs that are not on the requirements.txt .


## TFDS dataset tokenized

```
tfds build
```

You should manually fix the datasets import in transformers lib (because it has the same name as the tf.datsets and causes problem). (requirements for tfds build)


## Unigram models

Counting the frequency of each token on the [unigram_models.ipynb](./unigram_models.ipynb) . This is to calculate the perplexity normalized by the vocabulary for each LLM.
