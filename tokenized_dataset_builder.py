"""tokenized dataset."""

import tensorflow_datasets as tfds
import numpy as np
import dataclasses
from tensorflow_datasets.core.utils.lazy_imports_utils import datasets as hf_datasets
import transformers # Need to patch datasets import


@dataclasses.dataclass
class TokenizedConfig(tfds.core.BuilderConfig):
  language: str = 'english'
  kind: str = 'vanilla'
  dtype: np.dtype = np.uint16

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for tokenized dataset."""
  BUILDER_CONFIGS = [
        TokenizedConfig(name=f'{lang}_{kind}', language=lang, kind=kind, dtype=np.int32 if lang=='all' else np.uint16)
        for lang in ['arabic', 'azerbaijani', 'chinese', 'english', 'farsi', 'german', 'hebrew', 'hindi', 'korean', 'spanish', 'turkish', 'vietnamese']#, 'lim', 'all']
          for kind in ['vanilla', 'multi']
    ]
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    language = self.builder_config.language
    kind = self.builder_config.kind
    if kind == 'vanilla':
      shape = (None,)
    else:
      trans_table = np.loadtxt(f'./tokenizers/{language}_{kind}/multi.txt', dtype=np.int32)
      shape = (None, trans_table.shape[-1])
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'story': tfds.features.Tensor(shape=shape, dtype=self.builder_config.dtype),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None, #('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    language = self.builder_config.language
    kind = self.builder_config.kind
    tokenizer = transformers.AutoTokenizer.from_pretrained(f'./tokenizers/{language}_{kind}/', add_bos_token=True, add_eos_token=True)
    trans_table = None if kind == 'vanilla' else np.loadtxt(f'./tokenizers/{language}_{kind}/multi.txt', dtype=np.int32)
    dataset = hf_datasets.load_dataset("Gabrui/multilingual_TinyStories", language)
    train_val = dataset['train'].train_test_split(test_size=1024*8 if len(dataset['train'])>99999 else 512*8, seed=72)

    return {
        'train': self._generate_examples(train_val['train'], tokenizer, trans_table),
        'valid': self._generate_examples(train_val['test'], tokenizer, trans_table),
        'test': self._generate_examples(dataset['test'], tokenizer, trans_table),
    }

  def _generate_examples(self, dataset, tokenizer, trans_table):
    """Yields examples."""
    if trans_table is None:
      for i, data in enumerate(dataset):
        yield i, {
            'story': np.array(tokenizer(data["story"])['input_ids'], dtype=self.builder_config.dtype),
        }
    else:
      for i, data in enumerate(dataset):
        yield i, {
            'story': trans_table[tokenizer(data["story"])['input_ids']].astype(self.builder_config.dtype),
        }