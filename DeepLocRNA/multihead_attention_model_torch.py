import torch.nn as nn
import torch
from collections import OrderedDict
from hier_attention_mask_torch import Attention_mask
from hier_attention_mask_torch import QKVAttention
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import time
import gin
import inspect
from utils import *
from layers import *
import os


class DM3Loc_sequential(nn.Module):
    def __init__(self, drop_cnn, drop_flat, drop_input, pooling_size, fc_dim, nb_classes, dim_attention,
                 headnum, Att_regularizer_weight, normalizeatt, sharp_beta, attmod, W1_regularizer, 
                 activation, activation_att, attention, pool_type, cnn_scaler, att_type, input_dim, hidden, 
                 parnet_dim, pooling_opt, filter_length1, release_layers, prediction, fc_layer, mode, mfes, RNA_types, RNA_type, att, device):
                                                                                          
        super(DM3Loc_sequential, self).__init__()

        self.drop_cnn = drop_cnn
        self.drop_flat = drop_flat
        self.drop_input = drop_input
        self.pooling_size = pooling_size
        self.fc_dim = fc_dim
        self.nb_classes = nb_classes
        self.dim_attention = dim_attention
        self.activation = activation
        self.activation_att = activation_att
        self.attention = attention
        self.headnum = headnum
        self.Att_regularizer_weight = Att_regularizer_weight
        self.normalizeatt = normalizeatt
        self.sharp_beta = sharp_beta
        self.attmod = attmod
        self.W1_regularizer = W1_regularizer
        self.activation = activation
        self.activation_att = activation_att
        self.attention = attention
        self.pool_type = pool_type
        self.cnn_scaler = cnn_scaler
        self.att_type = att_type
        self.input_dim = input_dim
        self.hidden = hidden
        self.parnet_dim = parnet_dim
        self.pooling_opt = pooling_opt
        self.filter_length1 = filter_length1
        self.release_layers = release_layers
        self.prediction = prediction
        self.fc_layer = fc_layer
        self.mode = mode
        self.mfes = mfes
        self.RNA_types = RNA_types
        self.RNA_type = RNA_type
        self.att = att
        self.device = device
        

        encoding_seq_fold = {'(': [1, 0, 0, 0],
            ')': [0, 1, 0, 0],
            '.': [0, 0, 1, 0],
            'N': [0, 0, 0, 1]}
        embedding_vec_fold = np.array(list(encoding_seq_fold.values()), dtype=np.float32)

        encoding_seq = OrderedDict([
            ('UNK', [0, 0, 0, 0]),
            ('A', [1, 0, 0, 0]),
            ('C', [0, 1, 0, 0]),
            ('G', [0, 0, 1, 0]),
            ('T', [0, 0, 0, 1]),
            ('N', [0.25, 0.25, 0.25, 0.25])  # A or C or G or T
        ])
        
        
        #layer define
        self.dropout1 = nn.Dropout(drop_cnn)
        dropout1 = nn.Dropout(drop_cnn)
        self.dropout2 = nn.Dropout(drop_flat)
        dropout2 = nn.Dropout(drop_flat)
        self.dropout3 = nn.Dropout(drop_input)
        self.maxpool = nn.MaxPool1d(pooling_size, stride = pooling_size)
        maxpool = nn.MaxPool1d(pooling_size, stride = pooling_size)
        self.meanpool = nn.AvgPool1d(pooling_size, stride = pooling_size)
        self.maxpool_opt = nn.AvgPool1d(5, stride = 5)
        self.globalavgpool = nn.AdaptiveAvgPool1d(1)

        
        if attention == True:
            if att_type == "transformer":
                neurons = int(hidden*3*cnn_scaler/3)
            elif att_type == "self_attention":
                neurons = int(headnum*hidden*3*cnn_scaler/3)
        elif attention == False:
            if pooling_opt:
                neurons = int(1*hidden*3*cnn_scaler/3)
            else:
                neurons = int(1000*hidden*3*cnn_scaler/3)

        self.fc1 = nn.Linear(neurons, fc_dim)
        fc1 = nn.Linear(neurons, fc_dim)

        if self.mfes:
            fc_dim = fc_dim + 4
        self.fc2 = nn.Linear(fc_dim, nb_classes)
        fc2 = nn.Linear(fc_dim, nb_classes)
        fc3 = nn.Linear(neurons, nb_classes)
        self.fc3 = nn.Linear(neurons, nb_classes)

        #attention layers
        if att_type == "self_attention":
            self.Attention1 = Attention_mask(hidden=hidden, att_dim=dim_attention, r=headnum, activation= activation_att,return_attention=True, 
                                        attention_regularizer_weight=Att_regularizer_weight,normalize=normalizeatt,attmod=attmod,
                                        sharp_beta=sharp_beta)
            self.Attention2 = Attention_mask(hidden=hidden, att_dim=dim_attention, r=headnum, activation= activation_att,return_attention=True, 
                                        attention_regularizer_weight=Att_regularizer_weight,normalize=normalizeatt,attmod=attmod,
                                        sharp_beta=sharp_beta)


        elif att_type == "transformer":
            self.Attention1 = QKVAttention(hidden=hidden, att_dim=dim_attention, headnum=headnum)
            self.Attention2 = QKVAttention(hidden=hidden, att_dim=dim_attention, headnum=headnum)
            self.Attention3 = QKVAttention(hidden=hidden, att_dim=dim_attention, headnum=headnum)


        if mfes:
            new_encoding_obj = RNAembed(RNA_types = RNA_types)
            encoding_seq = new_encoding_obj()
            # print("new encoding_seq:", encoding_seq)
        embedding_vec = np.array(list(encoding_seq.values()), dtype=np.float32)

        self.embedding_layer = nn.Embedding(num_embeddings=len(embedding_vec),embedding_dim=len(embedding_vec[0]),_weight=torch.tensor(embedding_vec))
        self.embedding_layer_fold = nn.Embedding(num_embeddings=len(embedding_vec_fold),embedding_dim=len(embedding_vec_fold[0]),_weight=torch.tensor(embedding_vec_fold))
        self.myloss = MultiTaskLossWrapper()
        self.flatten = nn.Flatten()

        self.att1_A = None
        self.att2_A = None

        self.softmax = nn.Softmax()

        if fc_layer:
            self.FC_block = nn.Sequential(fc1,
                                        Actvation(activation),
                                        dropout2,
                                        fc2,
                                        nn.Sigmoid())
        else:
            self.FC_block = nn.Sequential(fc3,
                                        nn.Sigmoid())

        self.Parnet_block2 = nn.Sequential(Parnet_model(release_layers, prediction, device),
                                          maxpool,
                                          dropout1)#[256]
        
        self.Actvation = Actvation(activation)
             
        self.Pooling = Pooling(pool_type, pooling_size)
        

    def print_init_parameters(self):
        init_params = inspect.signature(self.__init__).parameters
        param_names = [param for param in init_params if param != 'self']
        for param_name in param_names:
            param_value = getattr(self, param_name)
            # print(f"{param_name}: {param_value}")

    def signal_preprocess(self, test, cutoff):
        test[test>cutoff] = 1
        test[test!=1] = 0
        return test
    def Att1(self, parnet_output, x_mask):   

        parnet_output = parnet_output*x_mask 
        parnet_output = torch.cat((parnet_output,x_mask), dim = 1)
        # print("parnet_output:",parnet_output.shape)
        att1,att1_A = self.Attention1(parnet_output, masks = True)
        
        self.att1_A = att1_A
        att1 = att1.transpose(1,2)
        
        return att1
    def Att2(self, parnet_output, x_mask):   

        parnet_output = parnet_output*x_mask 
        parnet_output = torch.cat((parnet_output,x_mask), dim = 1)
        # print("parnet_output:",parnet_output.shape)
        att2,att2_A = self.Attention2(parnet_output, masks = True)
        
        self.att2_A = att2_A
        att2 = att2.transpose(1,2)
        
        return att2

 

    def forward(self, x, x_mask, x_mfes=None):

        if not self.att:
            x = x.long()
            embedding_output = self.embedding_layer(x)#[8000, 4]
            embedding_output = embedding_output.transpose(1,2)#[4, 8000]         
            embedding_output = self.dropout3(embedding_output)
            embedding_output = embedding_output.to(torch.float32)
        else:
            embedding_output = x
        x_mask = x_mask.unsqueeze(1).float()
                    
        if self.mode == "feature":
            parnet_output = self.Parnet_block(embedding_output)#[256]
            return parnet_output
        elif self.mode == "full":
            if self.attention == True:
                parnet_output = self.Parnet_block2(embedding_output)#[256, 1000]
                output = self.Att1(parnet_output, x_mask) #[hidden, heads] 
                output = self.flatten(output)
                if x_mfes != None:
                    x_mfes = x_mfes.long()
                    embedding_output2 = self.embedding_layer(x_mfes[:,0])#n*4
                    output = self.fc1(output)
                    output = torch.cat((output, embedding_output2), dim=1)
                    output = Actvation(self.activation)(output)
                    output = self.dropout2(output)
                    output = self.fc2(output)
                    pred = nn.Sigmoid()(output)
                else:
                    pred = self.FC_block(output)
            else:
                parnet_output = self.Parnet_block2(embedding_output)#[256, 1000]
                pred = self.FC_block(parnet_output)#[7]
            return pred
    def mask_func(x):
        return x[0] * x[1]

@gin.configurable
class myModel1(pl.LightningModule):
    def __init__(self, drop_cnn, drop_flat, drop_input, pooling_size, fc_dim, nb_classes, dim_attention,
                 headnum, Att_regularizer_weight, normalizeatt, sharp_beta, attmod, W1_regularizer, 
                 activation, activation_att, attention, pool_type, cnn_scaler, att_type, input_dim, hidden, 
                 parnet_dim, pooling_opt, filter_length1, release_layers, prediction, fc_layer, mode, mfes,
                 lr, gradient_clip, class_weights, optimizer, weight_decay, OHEM, loss_type, add_neg, focal, RNA_type, att, species):
        super(myModel1, self).__init__()
        #only used to extract the RNA types
        current_path = os.getcwd()
        pj=lambda *path: os.path.abspath(os.path.join(*path))
        root_dir=os.getcwd()
        if species == "Human":
            dataset_get_types = pj(root_dir, "data", "allRNA", "allRNA_all_human_data_seq_mergedm3locall2_deduplicated2_filtermilnc.fasta")
        elif species == "Mouse":
            dataset_get_types = pj(root_dir, "data", "allRNA", "allRNA_all_mouse_data_seq_deduplicated.fasta")
        # dataset_get_types = os.path.join(current_path, "data", "allRNA", "allRNA_all_human_data_seq_mergedm3locall2_deduplicated2_filtermilnc.fasta")
        RNA_types = GetRNAtype(dataset = dataset_get_types)
        self.network = DM3Loc_sequential(drop_cnn, drop_flat, drop_input, pooling_size, fc_dim, nb_classes, dim_attention,
                 headnum, Att_regularizer_weight, normalizeatt, sharp_beta, attmod, W1_regularizer, 
                 activation, activation_att, attention, pool_type, cnn_scaler, att_type, input_dim, 
                 hidden, parnet_dim, pooling_opt, filter_length1, release_layers, prediction, fc_layer, mode, mfes, RNA_types, RNA_type, att, self.device)
        self.network = self.network.to(self.device)
        self.network = self.network.to(torch.float32)
        self.lr = lr
        self.weight_decay = weight_decay
        self.cnn_scaler = cnn_scaler
        self.gradient_clip = gradient_clip
        self.class_weights = class_weights 
        self.loss_fn = nn.BCELoss()
        self.optimizer_cls = eval(optimizer)
        self.train_loss = []
        self.val_binary_acc = []
        self.val_Multilabel_acc = []
        self.attention = attention
        self.att_type = att_type 
        self.optim = optimizer.split(".")[-1]
        self.mfes = mfes
        self.OHEM = OHEM
        self.keep_num = 10
        self.loss_type = loss_type
        self.nb_classes = nb_classes
        self.add_neg = add_neg
        self.focal = focal
        
    def weighted_binary_cross_entropy(self, output, target):

        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
        return torch.neg(torch.mean(loss))

    def naive_loss(self, y_pred, y_true, ohem=False,focal=False):

        loss_weight_ = self.class_weights
        loss_weight = []
        for i in range(self.nb_classes):
        # initialize weights
            loss_weight.append(torch.tensor(loss_weight_[i],requires_grad=False, device=self.device))


        num_task = y_true.shape[-1]
        num_examples = y_true.shape[0]
        k = 0.7

        def binary_cross_entropy(x, y,focal=True):
            alpha = 0.75
            gamma = 2

            pt = x * y + (1 - x) * (1 - y)
            at = alpha * y + (1 - alpha)* (1 - y)

            # focal loss
            if focal:
                loss = -at*(1-pt)**(gamma)*(torch.log(x) * y + torch.log(1 - x) * (1 - y))
            else:
                epsilon = 1e-4  # Small epsilon value
                # Add epsilon to x to prevent taking the logarithm of 0
                x = torch.clamp(x, epsilon, 1 - epsilon)
                loss = -(torch.log(x) * y + torch.log(1 - x) * (1 - y))
                # print("loss:", loss)
            return loss
        loss_output = torch.zeros(num_examples).to(device = self.device)
        for i in range(num_task):
            if loss_weight != None:
                out = loss_weight[i]*binary_cross_entropy(y_pred[:,i],y_true[:,i],focal)
                loss_output += out
            else:
                loss_output += binary_cross_entropy(y_pred[:, i],y_true[:,i],focal)

        # Online Hard Example Mining
        if ohem:
            val, idx = torch.topk(loss_output,int(0.7*num_examples))
            loss_output[loss_output<val[-1]] = 0
        loss = torch.sum(loss_output)/num_examples
        return loss
    def binary_accuracy(self, y_pred, y_true):
        # Round the predicted values to 0 or 1
        y_pred_rounded = torch.round(y_pred)
        # Calculate the number of correct predictions
        correct = (y_pred_rounded == y_true).float().sum()
        # Calculate the accuracy
        accuracy = correct / y_true.numel()
        return accuracy
    
    def categorical_accuracy(self, y_pred, y_true):
        # Get the index of the maximum value (predicted class) along the second dimension
        y_pred = torch.argmax(y_pred, dim=1)
        y_true = torch.argmax(y_true, dim=1)
        # Compare the predicted class with the target class and calculate the mean accuracy
        return (y_pred == y_true).float().mean()

    def forward(self, x, mask, x_mfe=None):
        x = x.to(self.device)
        mask = mask.to(self.device)
        if x_mfe != None:
            x_mfe = x_mfe.to(self.device)
        pred = self.network(x, mask, x_mfe)
        return pred
    
    def configure_optimizers(self):
        # optimizer = self.optimizer_cls(self.parameters(), lr = self.lr, weight_decay = 5e-5)
        if self.optim == "Adam":
            optimizer = self.optimizer_cls(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        elif self.optim == "SGD":
            optimizer =  self.optimizer_cls(self.parameters(), lr = self.lr, momentum=0.9, weight_decay = self.weight_decay)
        elif self.optim == "RMSPROP":
            optimizer =  self.optimizer_cls(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        return optimizer

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        print(f"Epoch duration: {epoch_time:.2f} seconds")
    def _attention_regularizer(self, attention):
        batch_size = attention.shape[0]
        headnum = self.network.headnum
        identity = torch.eye(headnum).to(self.device)  # [r,r]
        temp = torch.bmm(attention, attention.transpose(1, 2)) - identity  # [none, r, r]
        penal = 0.001 * torch.sum(temp**2) / batch_size
        return penal

    def training_step(self, batch, batch_idx, **kwargs):


        if self.mfes:
            x, x_mfes, mask, y= batch
            y = y.to(torch.float32)
            y_pred = self.forward(x, mask, x_mfes)
        else:
            if self.add_neg:
                x, mask, y= batch
                batch_size = x.shape[0]
                x_n = torch.tensor([neg_gen(i*(batch_idx+1),4000,4000,"seq")  for i in range(x.size()[0])], dtype = torch.float32).to(device = self.device)
                y_n = torch.tensor([neg_gen(i*(batch_idx+1),4000,4000,"y")  for i in range(y.size()[0])], dtype = torch.float32).to(device = self.device)
                mask_n = torch.tensor([neg_gen(i*(batch_idx+1),4000,4000,"mask")  for i in range(mask.size()[0])], dtype = torch.float32).to(device = self.device)

                x = torch.cat([x, x_n], axis = 0)
                y = torch.cat([y, y_n], axis = 0)
                mask = torch.cat([mask, mask_n], axis = 0)
                y_pred = self.forward(x, mask)
                # print("x", x)
                # print("y", y)
                # print("mask", mask)
                # print("y_pred", y_pred)
            else:
                x, mask, y= batch
                # print("x shape 0:", x.shape[0])
                y = y.to(torch.float32)
                y_pred = self.forward(x, mask)

        if self.loss_type == "learnable":
            # loss = self.naive_loss(y_pred, y)
            loss = self.MutiTaskLoss(y_pred,y)
        elif self.loss_type == "BCE":
            loss = self.loss_fn(y_pred, y)
        elif self.loss_type == "fixed_weight":

            loss = self.naive_loss(y_pred, y, ohem=self.OHEM, focal = self.focal)

        #Using the gradient clip to protect from gradient exploration
        if self.gradient_clip:
            nn.utils.clip_grad_norm_(self.network.parameters(), 1)

        l1_regularization = torch.tensor(0., device=self.device)
        for name, param in self.network.named_parameters(): 
            if 'Attention1.W1' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention1.W2' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W1' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W2' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W1' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W2' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention1.W_q' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention1.W_k' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention1.W_v' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention1.W_o' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W_q' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W_k' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W_v' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W_o' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W_q' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W_k' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W_v' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W_o' in name:
                l1_regularization += torch.norm(param, p=1)
            

        if self.attention and self.att_type == "self_attention":

            loss += l1_regularization*0.001
            if self.cnn_scaler == 1:
                loss += self._attention_regularizer(torch.transpose(self.network.att1_A, 1, 2))
            elif self.cnn_scaler == 2:
                loss += self._attention_regularizer(torch.transpose(self.network.att1_A, 1, 2))
                loss += self._attention_regularizer(torch.transpose(self.network.att2_A, 1, 2))
            elif self.cnn_scaler == 3:
                loss += self._attention_regularizer(torch.transpose(self.network.att1_A, 1, 2))
                loss += self._attention_regularizer(torch.transpose(self.network.att2_A, 1, 2))
                loss += self._attention_regularizer(torch.transpose(self.network.att3_A, 1, 2))
        if self.attention and self.att_type == "transformer":
            loss += l1_regularization*0.001
            if self.cnn_scaler == 1:
                loss += self._attention_regularizer(self.network.att1_A)
            elif self.cnn_scaler == 2:
                loss += self._attention_regularizer(self.network.att1_A)
                loss += self._attention_regularizer(self.network.att2_A)
            elif self.cnn_scaler == 3:
                loss += self._attention_regularizer(self.network.att1_A)
                loss += self._attention_regularizer(self.network.att2_A)    
                loss += self._attention_regularizer(self.network.att3_A)
            
  
        self.log("train_loss", loss, on_epoch = True, on_step = True)
        categorical_accuracy = self.categorical_accuracy(y_pred, y)
        categorical_accuracy_strict = self.categorical_accuracy_strict(y_pred, y)
        binary_accuracy = self.binary_accuracy(y_pred, y)
        
        self.log('train categorical_accuracy', categorical_accuracy, on_step = True, on_epoch=True, prog_bar = True)
        self.log('train categorical_accuracy_strict', categorical_accuracy_strict, on_step = True, on_epoch=True, prog_bar = True)
        self.log('train binary_accuracy', binary_accuracy, on_step = True, on_epoch=True, prog_bar = True)
 

        return loss
    def categorical_accuracy_strict(self, y_pred, y_true):
    # Find the index of the maximum value in each row (i.e., the predicted class)
        y_pred_class = torch.round(y_pred)
        com = y_pred_class == y_true
        correct = com.all(dim=1).sum()
        sample_num = y_true.size(0)
        accuracy = correct / sample_num
        return accuracy
    def validation_step(self, batch, batch_idx):
        if self.mfes:
            x, x_mfes, mask, y= batch
            y = y.to(torch.float32)
            y_pred = self.forward(x, mask, x_mfes)
        else:
            if self.add_neg:
                x, mask, y= batch
                x_n = torch.tensor([neg_gen(i*(batch_idx+1),4000,4000,"seq")  for i in range(x.size()[0])], dtype = torch.float32).to(device = self.device)
                y_n = torch.tensor([neg_gen(i*(batch_idx+1),4000,4000,"y")  for i in range(y.size()[0])], dtype = torch.float32).to(device = self.device)
                mask_n = torch.tensor([neg_gen(i*(batch_idx+1),4000,4000,"mask")  for i in range(mask.size()[0])], dtype = torch.float32).to(device = self.device)

                x = torch.cat([x, x_n], axis = 0)
                y = torch.cat([y, y_n], axis = 0)
                mask = torch.cat([mask, mask_n], axis = 0)
                # print("shape of adding negative:", x.shape)

                y_pred = self.forward(x, mask)

            else:
                x, mask, y= batch
                y = y.to(torch.float32)
                y_pred = self.forward(x, mask)

        categorical_accuracy = self.categorical_accuracy(y_pred, y)
        categorical_accuracy_strict = self.categorical_accuracy_strict(y_pred, y)
        binary_accuracy = self.binary_accuracy(y_pred, y)

        self.log('val categorical_accuracy', categorical_accuracy, on_step = True, on_epoch=True, prog_bar = True)
        self.log('val categorical_accuracy_strict', categorical_accuracy_strict, on_step = True, on_epoch=True, prog_bar = True)
        self.log('val binary_accuracy', binary_accuracy, on_step = True, on_epoch=True, prog_bar = True)

        if self.loss_type == "learnable":
            loss = self.MutiTaskLoss(y_pred,y)
        elif self.loss_type == "BCE":
            loss = self.loss_fn(y_pred, y)
        elif self.loss_type == "fixed_weight":
            loss = self.naive_loss(y_pred, y, ohem=self.OHEM, focal = False)
        l1_regularization = torch.tensor(0., device = self.device)
        for name, param in self.network.named_parameters(): 
            if 'Attention1.W1' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention1.W2' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W1' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W2' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W1' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W2' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention1.W_q' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention1.W_k' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention1.W_v' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention1.W_o' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W_q' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W_k' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W_v' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W_o' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W_q' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W_k' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W_v' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W_o' in name:
                l1_regularization += torch.norm(param, p=1)
            

        if self.attention and self.att_type == "self_attention":

            loss += l1_regularization*0.001
            #add the Attention regulizer
            if self.cnn_scaler == 1:
                loss += self._attention_regularizer(torch.transpose(self.network.att1_A, 1, 2))
            elif self.cnn_scaler == 2:
                loss += self._attention_regularizer(torch.transpose(self.network.att1_A, 1, 2))
                loss += self._attention_regularizer(torch.transpose(self.network.att2_A, 1, 2))
            elif self.cnn_scaler == 3:
                loss += self._attention_regularizer(torch.transpose(self.network.att1_A, 1, 2))
                loss += self._attention_regularizer(torch.transpose(self.network.att2_A, 1, 2))
                loss += self._attention_regularizer(torch.transpose(self.network.att3_A, 1, 2))
        if self.attention and self.att_type == "transformer":
            loss += l1_regularization*0.001
            if self.cnn_scaler == 1:
                loss += self._attention_regularizer(self.network.att1_A)
            elif self.cnn_scaler == 2:
                loss += self._attention_regularizer(self.network.att1_A)
                loss += self._attention_regularizer(self.network.att2_A)
            elif self.cnn_scaler == 3:
                loss += self._attention_regularizer(self.network.att1_A)
                loss += self._attention_regularizer(self.network.att2_A)    
                loss += self._attention_regularizer(self.network.att3_A)

        self.log("val_loss", loss, on_epoch = True, on_step = True)

        return {"categorical_accuracy": categorical_accuracy, "categorical_accuracy_strict":categorical_accuracy_strict,
                "binary_accuracy": binary_accuracy}
    def print_init_parameters(self):
        init_params = inspect.signature(self.__init__).parameters
        param_names = [param for param in init_params if param != 'self']
        for param_name in param_names:
            param_value = getattr(self, param_name)
            print(f"{param_name}: {param_value}")
