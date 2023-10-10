# %%
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution, LayerIntegratedGradients
from captum.attr import visualization as viz
import os, sys
import numpy as np
from PIL import Image
import os
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score,roc_auc_score,accuracy_score,matthews_corrcoef
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
import h5py
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import pickle
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
from torchsummary import summary
import sys
sys.path.insert(0, "../../")
sys.path.insert(0, "../../new_data")
sys.path.insert(0, "../../attention_weight_analysis")
from fine_tuning_deeprbploc_allRNA import *
from multihead_attention_model_torch_sequential_modifymuchpossible import *
import logomaker
import gin
from attention_weight_analysis import weight_analysis
from torch.utils.data import DataLoader, ConcatDataset
import time
import argparse
import os
# %% [markdown]
# # Loading the best model

# %%
class motif_analysis:
    def __init__(self, device = "cpu", fold = 0, layer = 20, batch_size = 4, dataset = None, test_mode = True):
        self.device = device
        self.fold = fold
        self.layer = layer
        self.batch_size = batch_size
        self.dataset = dataset
        self.test_mode = test_mode

    def load_model(self):
        ckpt_path = "/home/sxr280/DeepRBPLoc/Results/allRNA_fineruning/checkpoints_allRNA_False_%s_%s_human_True_4158_best.ckpt" % (self.layer, self.fold)
        gin.parse_config_file('/home/sxr280/DeepRBPLoc/Multihead_train_torch_sequential_modifymuchpossible.gin')
        hyperparams_1 = {
                        'fc_dim': 500,
                        'weight_decay': 1e-5,
                        'attention': True,####need to be changed
                        'lr': 0.0005,
                        'drop_flat':0.4,
                        'drop_cnn': 0.3,
                        'drop_input': 0.3,
                        'hidden':256, #256
                        'pooling_opt':True,
                        'filter_length1':3,
                        'activation':"gelu",
                        'optimizer':"torch.optim.Adam",
                        'release_layers': 20,#
                        'prediction':False,
                        'fc_layer' : True,
                        'cnn_scaler': 1,
                        'headnum': 3,
                        'mode' : "full",
                        'mfes' : True, ###
                        'OHEM' : False, ###
                        'loss_type': "BCE",
                        'class_weights': None,
                        'gradient_clip': True,##
                        "add_neg" : False,
                        'focal' : False,
                        "nb_classes": 9,
                        "dataset" : self.dataset,
                        "RNA_type" : "allRNA",
                        "dim_attention" : 50,
                        "prediction" : True,
                        "att" : True ###this is important
                        }
       
        

        model = myModel1(**hyperparams_1)
        model = model.to(device = self.device)
        checkpoint = torch.load(ckpt_path, map_location = self.device)
        model_state = checkpoint['state_dict']
        model.load_state_dict(model_state)
        model = model.eval()
        return model
    def load_data(self):
        obj = lncTune(load_data = True, flatten_tag = True, num_task = 9)
        Xtrain,Ytrain,Train_mask_label,Xtest,Ytest,Test_mask_label,Xval,Yval,Val_mask_label, encoding_keys, encoding_vectors = preprocess_data(left = 4000, right = 4000, dataset = self.dataset, padmod = "after", pooling_size=8, foldnum=5, pooling=True, RNA_type = "allRNA", RNA_tag = False)
        Xtrain_tag,Ytrain,Train_mask_label,Xtest_tag,Ytest,Test_mask_label,Xval_tag,Yval,Val_mask_label, encoding_keys, encoding_vectors = preprocess_data(left = 4000, right = 4000, dataset = self.dataset, padmod = "after", pooling_size=8, foldnum=5, pooling=True, RNA_type = "allRNA", RNA_tag = True)


        X_train = torch.from_numpy(Xtrain[self.fold])#.to(self.device, torch.float)
        X_train_tag = torch.from_numpy(Xtrain_tag[self.fold])#.to(self.device, torch.float)
        Y_train = torch.from_numpy(Ytrain[self.fold])#.to(self.device, torch.float)
        Train_mask = torch.from_numpy(Train_mask_label[self.fold])#.to(self.device, torch.float)
                 
        train_dataset = torch.utils.data.TensorDataset(X_train, X_train_tag, Train_mask, Y_train)
        dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size = self.batch_size, shuffle = False)

        #validation data processes
        X_val = torch.from_numpy(Xval[self.fold])#.to(self.device, torch.float)
        X_val_tag = torch.from_numpy(Xval_tag[self.fold])#.to(self.device, torch.float)
        Y_val = torch.from_numpy(Yval[self.fold])#.to(self.device, torch.float)
        Val_mask = torch.from_numpy(Val_mask_label[self.fold])#.to(self.device, torch.float)

        val_dataset = torch.utils.data.TensorDataset(X_val, X_val_tag, Val_mask, Y_val)
        dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size = self.batch_size, shuffle = False)

        #test data process
        X_test = torch.from_numpy(Xtest[self.fold])#.to(self.device, torch.float)
        X_test_tag = torch.from_numpy(Xtest_tag[self.fold])#.to(self.device, torch.float)
        Y_test = torch.from_numpy(Ytest[self.fold])#.to(self.device, torch.float)
        Test_mask = torch.from_numpy(Test_mask_label[self.fold])#.to(self.device, torch.float)
        test_dataset = torch.utils.data.TensorDataset(X_test, X_test_tag, Test_mask, Y_test)
        
        dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size = self.batch_size, shuffle = False)
    

        return dataloader_train, dataloader_val, dataloader_test
    def IG_calculation(self, target, n_steps = 50):
        model = self.load_model()
        dataloader_train, dataloader_val, dataloader_test = self.load_data()
        #concatenate the dataloader together
        combined_dataset = ConcatDataset([dataloader_train.dataset, dataloader_val.dataset, dataloader_test.dataset])
        combined_dataloader = DataLoader(combined_dataset, batch_size=self.batch_size , shuffle=False)
        # combined_dataloader = combined_dataloader.to(self.device)
        ig = IntegratedGradients(model)
        all_att = torch.tensor([]).to(self.device)
        X_all = torch.tensor([]).to(self.device)
        #getting the fold data
        dataname = os.path.abspath(self.dataset).split(".fasta")[0]
        # print(dataname)
        # print(os.path.join(dataname, "/%s5%s.pkl" % ("Train", self.fold)))
        Train = pickle.load(open(dataname + "/%s5%s.pkl" % ("Train", self.fold), "rb"))
        Test = pickle.load(open(dataname + "/%s5%s.pkl" % ("Test", self.fold), "rb"))
        Val = pickle.load(open(dataname + "/%s5%s.pkl" % ("Val", self.fold), "rb"))
        all_id = Train + Val + Test
        if not self.test_mode:
            for i, batch in enumerate(combined_dataloader):
                print("Running the batch:", i)
                X, X_tag, X_mask, y = batch
                X = X.to(self.device)
                X_tag = X_tag.to(self.device)
                X_mask = X_mask.to(self.device)
                #processing the input after embedding
                X = X.long()
                # print("X device:", X.device)
                embedding_output = model.network.embedding_layer(X)#[8000, 4]
                embedding_output = embedding_output.transpose(1,2)#[4, 8000]         
                embedding_output = embedding_output.to(torch.float32)

                # Calculate attributions for the embedded input tensor
                attributions = ig.attribute(inputs=(embedding_output), target=target, n_steps=n_steps, additional_forward_args = (X_mask,X_tag), return_convergence_delta = False, internal_batch_size = 8)

                all_att = torch.cat((all_att,attributions), 0)
                X_all = torch.cat((X_all,X), 0)
        else:
            X, X_tag, X_mask, y = next(iter(combined_dataloader))
            X = X.to(self.device)
            X_tag = X_tag.to(self.device)
            X_mask = X_mask.to(self.device)
            #processing the input after embedding
            X = X.long()
            # print("X device:", X.device)
            embedding_output = model.network.embedding_layer(X)#[8000, 4]
            embedding_output = embedding_output.transpose(1,2)#[4, 8000]         
            embedding_output = embedding_output.to(torch.float32)

            # Calculate attributions for the embedded input tensor
            attributions = ig.attribute(inputs=(embedding_output), target=target, n_steps=n_steps, additional_forward_args = (X_mask,X_tag), return_convergence_delta = False, internal_batch_size = 8)

            all_att = torch.cat((all_att,attributions), 0)
            X_all = torch.cat((X_all,X), 0)
            all_id  = all_id[:self.batch_size]
        #detach the torch to numpy
        all_att = all_att.permute(0, 2, 1).cpu().detach().numpy()
        

        return all_att, X_all, all_id


# # Choose the best 5-mer across the sequences

# %%
from collections import OrderedDict
def max_fuc(att, window):
    max_p = 0
    max_v = 0
    for p,v in enumerate(att):
        v_sum = sum(att[p:min((p+window), len(att))])
        # print(v_sum)
        if v_sum > max_v:
            max_p = p
            max_v = v_sum
    # print("max_v", max_v)
    return max_v, max_p

def get_max_motif(all_att, X_all, all_id, window, prefix):
    flat_att = all_att.sum(axis=2)
    print("flat_att", flat_att)
    flat_att_p = [max_fuc(list(s), window)[1] for s in flat_att]
    flat_att_val = [max_fuc(list(s), window)[0] for s in flat_att]
    #from p to 5-mer
    motifs_num = [s[flat_att_p[idx]:(flat_att_p[idx]+window)] for idx,s in enumerate(X_all)]

    encoding_seq = OrderedDict([
                ('UNK', [0, 0, 0, 0]),
                ('A', [1, 0, 0, 0]),
                ('C', [0, 1, 0, 0]),
                ('G', [0, 0, 1, 0]),
                ('T', [0, 0, 0, 1]),
                ('N', [0.25, 0.25, 0.25, 0.25])  # A or C or G or T
            ])
    seq_encoding_keys = list(encoding_seq.keys())
    motifs = list(map(lambda x: [seq_encoding_keys[int(i)] for i in list(x)], motifs_num))# maybe apply or map
    out_str = ""
    for idx,item in enumerate(all_id):
        print(item)
        print(motifs[idx])
        print(flat_att_val[idx])
        line = item + "\t" + "".join(motifs[idx]) + "\t" + str(flat_att_val[idx]) + "\n"
        out_str += line
    with open("./%s_df.txt" % prefix, "w") as f1:
        f1.write(out_str)

    #saving the att value in the 
    pickle.dump(flat_att, open("./%s_att_value.pkl" % prefix, "wb"))
    #saving all the all_att
    pickle.dump(all_att, open("./%s_4dimsatt_value.pkl" % prefix, "wb"))
    #saving the input data
    pickle.dump(X_all, open("./%s_X_all.pkl" % prefix, "wb"))
    #saing the alignment
    pickle.dump(encoding_seq, open("./%s_encoding_weight.pkl" % prefix, "wb"))
    return motifs, flat_att_p, flat_att_val


def main(k_mer=5, target = 0, test_mode=True):
    target_names = ["Nucleus","Exosome","Cytosol","Cytoplasm","Ribosome","Membrane","Endoplasmic_reticulum", "Microvesicle", "Mitochondrion"]
    target_name = target_names[target]

    #getting the proper input dataset
    # dataset = "/home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2_filtermilnc.fasta"
    dataset = "./data_%s.fasta" % target_name

    #creating the file path according to the fasta file
    # Check if the directory was created
    create_path = os.path.abspath(dataset).split(".fasta")[0]
    
    if os.path.exists(create_path):
        print(f"Directory '{create_path}' has already created.")
    else:
        print(f"creating directory '{create_path}'.")
        os.makedirs(create_path)
    
    
    ana = motif_analysis(fold = 3, device = "cuda", layer = 20, batch_size = 32, dataset = dataset, test_mode = test_mode)
    all_att,X_all,all_id = ana.IG_calculation(target = target, n_steps = 25)
    prefix = "kmer_" + str(k_mer) + "target_" + str(target)
    motifs, flat_att_p, flat_att_val = get_max_motif(all_att, X_all, all_id, k_mer, prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=3, help='which compartment you want to calculate')
    # parser.add_argument('--t', type=int, default=3, help='which compartment you want to calculate')
    
    args = parser.parse_args()
    start_time = time.time()
    print("running target:", args.target)
    main(k_mer=5, target = args.target, test_mode = False)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")


# %%


