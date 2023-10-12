# %%
import gin
import torch
import torchmetrics

# %%
def softmax(x, dim=-1):
    return torch.softmax(x, dim=dim)

# %%
def pearson_corrcoef(x, y, dim=-1):
    x = x - torch.unsqueeze(torch.mean(x, dim), dim)
    y = y - torch.unsqueeze(torch.mean(y, dim), dim)
    return torch.sum(x * y, dim) / torch.sqrt(torch.sum(x ** 2, dim) * torch.sum(y ** 2, dim))

# %%
@gin.configurable()
class PearsonCorrCoeff(torchmetrics.MeanMetric):
    def __init__(self, dim=-1, postproc_fn=softmax, reduction=torch.mean, *args, **kwargs):
        super(PearsonCorrCoeff,  self).__init__(*args, **kwargs)

        self.dim = dim
        self.postproc_fn = postproc_fn
        self.reduction = reduction

    def update(self, y: torch.Tensor, y_pred: torch.Tensor):
        assert y.shape == y_pred.shape

        if self.postproc_fn is not None:
            y_pred = self.postproc_fn(y_pred)
        
        pcc = pearson_corrcoef(y, y_pred, dim=self.dim)
        pcc = torch.nan_to_num(pcc, 0.0) # replace nan's with 0 (this might underestimate the pcc)

        reduced_pcc = self.reduction(pcc)

        # update (i.e. take mean)
        super().update(reduced_pcc)

# %%
@gin.configurable()
class FilteredPearsonCorrCoeff(torchmetrics.MeanMetric):
    def __init__(self, min_height=2, min_count=2, dim=-1, postproc_fn=softmax, *args, **kwargs):
        super(FilteredPearsonCorrCoeff,  self).__init__(*args, **kwargs)

        self.min_height = min_height
        self.min_count = min_count
        self.dim = dim
        self.postproc_fn = postproc_fn

    def update(self, y: torch.Tensor, y_pred: torch.Tensor):
        assert y.shape == y_pred.shape

        if self.postproc_fn is not None:
            y_pred = self.postproc_fn(y_pred)
        
        pcc = pearson_corrcoef(y, y_pred, dim=self.dim)
        mean_pcc = self.compute_mean(pcc, y)

        # update (i.e. take mean)
        super().update(mean_pcc)

    def compute_mean(self, values: torch.Tensor, y: torch.Tensor):
        # create boolean tensor of entries that are *not* NaNs
        values_is_not_nan_mask = torch.logical_not(torch.isnan(values))
        # convert nan's to 0
        values = torch.nan_to_num(values, 0.0)

        # check if required height is reached per experiment
        if self.min_height is not None:
            # should be shape (batch_size, experiments)
            y_min_height_mask = (torch.max(y, dim=-1).values >= self.min_height)
        
        # check if required count is reached per experiment
        if self.min_count is not None:
            # should be shape (batch_size, experiments)
            y_min_count_mask = (torch.sum(y, dim=-1) >= self.min_count)
        
        # boolean mask indicating which experiment (in each batch) passed nan, heigh and count (and is thus used for the final mean PCC)
        passed_boolean_mask = torch.sum(torch.stack([values_is_not_nan_mask, y_min_height_mask, y_min_count_mask]), dim=0) > 0

        # mask out (i.e. zero) all PCC values that did not pass
        values_masked = torch.mul(values, passed_boolean_mask.to(torch.float32))

        # compute mean by only dividing by #-elements that passed
        values_mean = torch.sum(values_masked)/torch.sum(passed_boolean_mask)

        # if ignore_nan:
        #     # only divide by #-elements not NaN
        #     values_mean = torch.sum(values)/torch.sum(values_is_not_nan)
        # else:
        #     values_mean = torch.mean(values)
        
        return values_mean
