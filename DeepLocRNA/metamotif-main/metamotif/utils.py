# %%
import numpy as np
import tensorflow as tf

# %%
base2int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def sequence2int(sequence):
    return [base2int.get(base, 999) for base in sequence]

def sequence2onehot(sequence):
    return tf.one_hot(sequence2int(sequence), depth=4)

# %%
def write_motif_tsv(motif_array, filepath, sigma=['A', 'C', 'G', 'U'], meta_info={}):
    assert motif_array.shape[1] == len(sigma)
    with open("/home/sxr280/DeepRBPLoc/motif_analysis/metamotif-main/output/" + filepath, 'w') as f:
        for key, value in meta_info.items():
            print(f'#{key}={value}', file=f)
        print('\t'.join(sigma), file=f)
        for row in motif_array:
            print('\t'.join(map(str, row)), file=f)