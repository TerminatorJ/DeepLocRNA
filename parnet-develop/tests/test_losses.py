# %%
import unittest

import torch
from torch.distributions import Multinomial
import tensorflow as tf
import tensorflow_probability as tfp

from parnet.losses import MultinomialNLLLossFromLogits, multinomial_nll_loss

# %%
def compute_manual_multinomial_nll(counts, logits):
    nll = []
    for i in range(counts.shape[0]):
        for j in range(counts.shape[2]):
            counts_ij, logits_ij = counts[i, :, j], logits[i, :, j]
            # print(Multinomial(total_count=torch.sum(single_y), logits=single_y_pred))
            nll.append(-Multinomial(int(torch.sum(counts_ij)), logits=logits_ij).log_prob(counts_ij))
    return torch.mean(torch.tensor(nll))

# %%
def compute_multinomial_nll_tensorflow(y, y_pred):
    y_tf, y_pred_tf = tf.constant(y, dtype=tf.float32), tf.constant(y_pred, dtype=tf.float32)
    return tf.reduce_mean(-1. * tfp.distributions.Multinomial(total_count=tf.reduce_sum(y_tf, axis=-1), logits=y_pred_tf).log_prob(y_tf))

# %%
class TestLosses(unittest.TestCase):

    def test_multinomial_nll_loss(self):
        # arrange
        y, y_pred = torch.randint(0, 10, size=(2, 7, 101)), torch.rand(2, 7, 101)
        nll_tf = torch.tensor(compute_multinomial_nll_tensorflow(y.numpy(), y_pred.numpy()).numpy(), dtype=torch.float32)

        # act
        nll = multinomial_nll_loss(y, y_pred)

        # assert
        try:
            assert torch.isclose(nll, nll_tf, atol=1e-6)
        except AssertionError:
            print(nll, nll_tf)
            raise

    def test_MultinomialNLLLossFromLogits(self):
        # arrange
        y, y_pred = torch.randint(0, 10, size=(2, 7, 101)), torch.rand(2, 7, 101)
        nll_tf = torch.tensor(compute_multinomial_nll_tensorflow(y.numpy(), y_pred.numpy()).numpy(), dtype=torch.float32)

        # act
        nll = MultinomialNLLLossFromLogits()(y, y_pred)

        # assert
        try:
            assert torch.isclose(nll, nll_tf, atol=1e-6)
        except AssertionError:
            print(nll, nll_tf)
            raise
