from multihead_attention_model_torch import *
import torch.nn as nn
import torch
import math
from collections import OrderedDict
import torch.nn.init as init

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class ParentEmbedFloat(nn.Module):
    def __init__(self):
        super(ParentEmbedFloat, self).__init__()

    def forward(self, x):
        x = x.to(device=self.device)
        # print("parnet output:", x, x.shape)
        return x.float()


class Pooling(nn.Module):
    def __init__(self, type, pooling_size):
        super(Pooling, self).__init__()
        self.type = type
        self.maxpool = nn.MaxPool1d(pooling_size, stride = pooling_size)
        self.meanpool = nn.AvgPool1d(pooling_size, stride = pooling_size)
        if self.type == "None":
            self.layer_name = "NoPooling"
        else:
            self.layer_name = f"{self.type}_pooling_{pooling_size}"
    def forward(self, x):
        if self.type == "max":
            x = self.maxpool(x)
        elif self.type == "mean":
            x = self.meanpool(x)
        elif self.type == "None":
            pass
        return x
    

    
class Actvation(nn.Module):
    def __init__(self, name):
        super(Actvation, self).__init__()
        self.name = name
        self.layer_name = None
    def gelu(self, input_tensor):
        """Gaussian Error Linear Unit.

        This is a smoother version of the RELU.
        Original paper: https://arxiv.org/abs/1606.08415

        Args:
            input_tensor: float Tensor to perform activation.

        Returns:
            `input_tensor` with the GELU activation applied.
        """
        cdf = 0.5 * (1.0 + torch.erf(input_tensor / math.sqrt(2.0)))
        return input_tensor * cdf

    def forward(self, x):
        if self.name == "relu":
            x = torch.nn.functional.relu(x)
            self.layer_name = "Activation_ReLU"
        elif self.name == "gelu":
            x = self.gelu(x)
            self.layer_name = "Activation_GeLU"
        elif self.name == "leaky":
            x = torch.nn.functional.leaky_relu(x)
            self.layer_name = "Activation_Leaky"

        return x
    
class Parnet_model(nn.Module):
    def __init__(self, release_layers, prediction):
        super(Parnet_model, self).__init__()
        self.release_layers = release_layers
        if prediction:
            self.parnet_model = torch.load("/home/sxr280/DeepRBPLoc/parnet_model/network.PanRBPNet.2023-03-13.ckpt", map_location=torch.device('cpu'))
        else:
            self.parnet_model = torch.load("/home/sxr280/DeepRBPLoc/parnet_model/network.PanRBPNet.2023-03-13.ckpt", map_location=torch.device("cuda"))
               
    def forward(self, x):
        x = x.to(torch.float32)
        freeze_index = len([i for i in self.parnet_model.named_parameters()]) - self.release_layers
        for i, (name, param) in enumerate(self.parnet_model.named_parameters()):
            if i < freeze_index:
                param = param.to(torch.float32)
                param.requires_grad = False
            else:
                param.requires_grad = True
        x = self.parnet_model.forward(x)#[256,8000]
        # x = x.float()
        return x    
    

class RNAembed(nn.Module):
    def __init__(self, RNA_types):
        super(RNAembed, self).__init__()
        self.RNA_types = RNA_types
        torch.manual_seed(42)
        self.RNA_embedding = nn.Embedding(len(set(RNA_types)), 4)
        init.uniform_(self.RNA_embedding.weight, a=0, b=1)
        self.encoding_seq = OrderedDict([
            ('UNK', [0, 0, 0, 0]),
            ('A', [1, 0, 0, 0]),
            ('C', [0, 1, 0, 0]),
            ('G', [0, 0, 1, 0]),
            ('T', [0, 0, 0, 1]),
            ('N', [0.25, 0.25, 0.25, 0.25])]) # A or C or G or T

    def forward(self):
        for RNA_type, embed in zip(self.RNA_types, (self.RNA_embedding.weight.tolist())):
            self.encoding_seq[RNA_type] = embed
        return self.encoding_seq


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, num_task=7):
        super(MultiTaskLossWrapper, self).__init__()
        self.num_task = num_task
        self.log_vars = nn.Parameter(torch.zeros((num_task)))
        # weights = torch.tensor([1,1,7,1,3,5,8])
        self.loss_fn = nn.BCELoss(weight = None)
    def binary_cross_entropy(self, x, y):
        epsilon = 1e-4
        x = torch.clamp(x, epsilon, 1 - epsilon)
        loss = -(torch.log(x) * y + torch.log(1 - x) * (1 - y))
        return torch.mean(loss)
    def forward(self, y_pred,targets):
        print("y_pred:", y_pred)
        print("targets:", targets)
        
        loss_output = 0
        for i in range(self.num_task):
            # print()
            out = torch.exp(-self.log_vars[i])*self.binary_cross_entropy(y_pred[:,i],targets[:,i]) + self.log_vars[i]
            print("out %s" % i, out)
            loss_output += out
        loss_output = loss_output/self.num_task
        print("loss_output", loss_output)

        return loss_output