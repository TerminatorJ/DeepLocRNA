# %%
import gin
import torch
import torch.nn as nn
import torchmetrics

# %%
def log_likelihood_from_logits(y, y_pred, dim):
    return torch.sum(torch.mul(torch.log_softmax(y_pred, dim=dim), y), dim=dim) + log_combinations(y, dim)

def log_combinations(input, dim):
    total_permutations = torch.lgamma(torch.sum(input, dim=dim) + 1)
    counts_factorial = torch.lgamma(input + 1)
    redundant_permutations = torch.sum(counts_factorial, dim=dim)
    return total_permutations - redundant_permutations

def multinomial_neg_log_probs(y, y_pred, dim=-1):
    return log_likelihood_from_logits(y, y_pred, dim) * -1

def multinomial_nll_loss(y, y_pred, dim=-1):
    return torch.mean(multinomial_neg_log_probs(y, y_pred, dim))

# %%
@gin.configurable()
class MultinomialNLLLossFromLogits(torchmetrics.MeanMetric):
    def __init__(self, dim=-1, reduction=torch.mean, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reduction = reduction
        self.dim = dim

    def update(self, y: torch.Tensor, y_pred: torch.Tensor):
        assert y_pred.shape == y.shape

        nll = multinomial_neg_log_probs(y, y_pred, dim=self.dim)
        assert nll.shape == y_pred.shape[:-1]

        # update running mean
        super().update(self.reduction(nll))


# %%
@gin.configurable()
class ClippedMultinomialNLLLossFromLogits(torchmetrics.MeanMetric):
    def __init__(self, dim=-1, reduction=torch.mean, clip=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reduction = reduction
        self.dim = dim

        assert clip >= 0
        self.clip = clip

    def update(self, y: torch.Tensor, y_pred: torch.Tensor):
        assert y_pred.shape == y.shape
        
        # clip y and y_pred (so 3'/5' ends are ignored during loss calculation)
        y_dim_size_before_clipping = y.shape[self.dim]
        y, y_pred = self.clip_tensor(y), self.clip_tensor(y_pred)

        # assert that clipping worked
        assert y.shape[self.dim] == y_pred.shape[self.dim] == y_dim_size_before_clipping-(2*self.clip)

        nll = multinomial_neg_log_probs(y, y_pred, dim=self.dim)
        assert nll.shape == y_pred.shape[:-1]

        # update running mean
        super().update(self.reduction(nll))
    
    def clip_tensor(self, tensor):
        return torch.narrow(tensor, self.dim, self.clip, tensor.shape[self.dim]-(2*self.clip))
