# %%
import sys

import gin
import torch
import tensorflow as tf

#from rbpnet import io

# %%
@gin.configurable(denylist=['filepath', 'features_filepath'])
class TFIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, filepath, features_filepath=None, batch_size=64, cache=True, shuffle=None):
        super(TFIterableDataset).__init__()
        
        # load tfrecord file and create tf.data pipeline 
        self.dataset = self._load_dataset(filepath, features_filepath, batch_size, cache, shuffle)

    def _load_dataset(self, filepath, features_filepath=None, batch_size=64, cache=True, shuffle=None):
        # no not serialize - only after shuffle/cache 
        dataset = io.dataset_ops.load_tfrecord(filepath, deserialize=False)
        if cache:
            dataset = dataset.cache()
        if shuffle:
            dataset = dataset.shuffle(shuffle)

        # deserialize proto to example
        if features_filepath is None:
            features_filepath = filepath + '.features.json'
        self.features = io.dataset_ops.features_from_json_file(features_filepath)
        dataset = io.dataset_ops.deserialize_dataset(dataset, self.features)

        # batch & prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # format example & prefetch
        dataset = dataset.map(self._format_example, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _format_example(self, example):
        # move channel dim from -1 to -2
        # example['inputs']['input'] = tf.transpose(example['inputs']['input'], perm=[0, 2, 1])
        # example['outputs']['signal']['total'] = tf.transpose(example['outputs']['signal']['total'], perm=[0, 2, 1])

        example = {
            'inputs': {
                'sequence': tf.transpose(example['inputs']['input'], perm=[0, 2, 1])},
            # 'outputs': {
            #     'total': tf.transpose(example['outputs']['signal']['total'], perm=[0, 2, 1]),
            #     'control': tf.transpose(example['outputs']['signal']['control'], perm=[0, 2, 1]),
            # },
            'outputs': {
                'total': example['outputs']['signal']['total'],
                'control': example['outputs']['signal']['control'],
            },
        }

        # return (input: Tensor, output: Tensor)
        return example
    
    def process_example(self, example):
        return example
    
    def _to_pytorch_compatible(self, example):
        return tf.nest.map_structure(lambda x: torch.tensor(x).to(torch.float32), example)

    def __iter__(self):
        for example in self.dataset.as_numpy_iterator():
            processed_pytorch_example = self._to_pytorch_compatible(self.process_example(example))
            yield processed_pytorch_example['inputs'], processed_pytorch_example['outputs']

# %%
@gin.configurable(denylist=['filepath', 'features_filepath'])
class MaskedTFIterableDataset(TFIterableDataset):
    def __init__(self, mask_filepaths=None, **kwargs):
        super(MaskedTFIterableDataset, self).__init__(**kwargs)
        self.composite_mask = None
        if mask_filepaths is not None:
            self.composite_mask = self._make_composite_mask(mask_filepaths)

    def _make_composite_mask(self, mask_filepaths):
        composite_mask = torch.load(mask_filepaths[0])
        for filepath in mask_filepaths[1:]:
            composite_mask = torch.logical_and(composite_mask, filepath)
        return composite_mask
    
    def mask_structure(self, structure, mask):
        try:
            return tf.nest.map_structure(lambda tensor: tensor[:, mask], structure)
        except:
            print(mask.shape, mask.dtype, file=sys.stderr)
            raise


    def process_example(self, example):
        example['outputs'] = self.mask_structure(example['outputs'], self.composite_mask)
        return example

# %%
@gin.configurable()
class MeanESMEmbeddingMaskedTFIterableDataset(MaskedTFIterableDataset):
    def __init__(self, embedding_matrix_filepath, masks=None, **kwargs):
        super(MeanESMEmbeddingMaskedTFIterableDataset, self).__init__(masks, **kwargs)
        self.embedding_matrix = torch.load(embedding_matrix_filepath)
    
    def process_example(self, example):
        # add protein embedding to inputs
        example['inputs']['embedding'] = self.embedding_matrix[self.composite_mask] if self.composite_mask is not None else self.embedding_matrix
        if self.composite_mask is not None:
            example['outputs'] = self.mask_structure(example['outputs'], self.composite_mask)
        return example

# %%
class DummyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, n) -> None:
        super(DummyIterableDataset).__init__()
        
        self.n = n

    def __iter__(self):
        for i in range(self.n):
            yield (torch.rand(16, 4, 101), torch.rand(16, 101, 7))
