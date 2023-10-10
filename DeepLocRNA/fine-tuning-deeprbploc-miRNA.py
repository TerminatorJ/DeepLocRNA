# %% [markdown]
# Based on our pre-training model, we can do the fine-tuning

# %%
import sys
sys.path.insert(0, "../")
from Multihead_train_torch_booster_modifymuchpossible import *
sys.path.insert(0, "./new_data")
import torch
from multihead_attention_model_torch_sequential_modifymuchpossible import *
import gin
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score,roc_auc_score,accuracy_score,matthews_corrcoef,f1_score
import time
from torchinfo import summary
from cross_RNA_prediction import LncValidation
import argparse
# dataset = "/home/sxr280/DeepRBPLoc/new_data/lncRNA/lncRNA_all_data_seq_filtered.fasta"
# dataset = "/home/sxr280/DeepRBPLoc/new_data/lncRNA/lncRNA_all_data_seq_includingmouse.fasta"

class lncTune:
    def __init__(self, left = 4000, right = 4000, device = "cuda", num_task = 5, fold = 0, load_weight = True):
        self.left = left
        self.right = right
        self.device = device
        self.num_task = num_task
        self.fold = fold
        self.load_weight = load_weight

    def get_binary(self, y):
        all_loc = ["Nucleus","Exosome","Cytosol","Cytoplasm","Ribosome","Membrane","Endoplasmic reticulum","Microvesicle", "Mitochondrion"]
        extra_idx = [all_loc.index(i) for i in ["Microvesicle","Exosome"]]
        print(extra_idx)
        intra_idx = [all_loc.index(j) for j in ["Nucleus","Cytosol","Cytoplasm","Ribosome","Membrane","Endoplasmic reticulum","Mitochondrion"]]
        print(intra_idx)
        extra_sum = y[:,extra_idx].sum(axis=1)
        print(extra_sum)
        intra_sum = y[:,intra_idx].sum(axis=1)
        print(intra_sum)
        new_y = np.column_stack((intra_sum, extra_sum))
        new_y = np.array(new_y, dtype = "bool")
        new_y = np.array(new_y, dtype = "int")
        print("new_y", new_y)
        return new_y
    def getdata(self, dataset=None, batch_size = 64):
        Xtrain,Ytrain,Train_mask_label,Xtest,Ytest,Test_mask_label,Xval,Yval,Val_mask_label, encoding_keys, encoding_vectors = preprocess_data(left = self.left, right = self.right, dataset = dataset, padmod = "after", pooling_size=8, foldnum=5, pooling=True)
        print("running fold:", self.fold)
        print("loading weight? ", self.load_weight)
        #For training dataset
        X_train = Xtrain[self.fold]
        # print("X_train", X_train[0], X_train[1])
        print("X_train shape:", X_train.shape)
        Y_train = Ytrain[self.fold][:,[0,1,3,7,8]]#only select 5 compartments
        # Y_train = self.get_binary(Ytrain[self.fold])
        Train_mask = Train_mask_label[self.fold]
        print("Train_mask", Train_mask.shape)
        #For validation dataset
        X_val = Xval[self.fold]
        Y_val = Yval[self.fold][:,[0,1,3,7,8]]#only select 5 compartments
        # Y_val = self.get_binary(Yval[self.fold])
        Val_mask = Val_mask_label[self.fold]

        X_test = Xtest[self.fold]
        Y_test = Ytest[self.fold][:,[0,1,3,7,8]]#only select 5 compartments
        # Y_test = self.get_binary(Ytest[self.fold])
        Test_mask = Test_mask_label[self.fold]


        
        weight_dict = self.cal_loss_weight(Y_train, beta=0.99999)
        ratio = np.sum(Y_train, axis=0)/Y_train.shape[0]
        print("The ratio of each compartment:", ratio, np.sum(Y_train, axis=0))
        print("The test data of each compartment:", ratio, np.sum(Y_test, axis=0))
        #training data processes
        X_train = torch.from_numpy(X_train).to(self.device, torch.float)
        Y_train = torch.from_numpy(Y_train).to(self.device, torch.float)
        Train_mask = torch.from_numpy(Train_mask).to(self.device, torch.float)
        train_dataset = torch.utils.data.TensorDataset(X_train, Train_mask, Y_train)
        dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

        #validation data processes
        X_val = torch.from_numpy(X_val).to(self.device, torch.float)
        Y_val = torch.from_numpy(Y_val).to(self.device, torch.float)
        Val_mask = torch.from_numpy(Val_mask).to(self.device, torch.float)
        val_dataset = torch.utils.data.TensorDataset(X_val, Val_mask, Y_val)
        dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
        
        #test data process
        X_test = torch.from_numpy(X_test).to(self.device, torch.float)
        Y_test = torch.from_numpy(Y_test).to(self.device, torch.float)
        Test_mask = torch.from_numpy(Test_mask).to(self.device, torch.float)
        test_dataset = torch.utils.data.TensorDataset(X_test, Test_mask, Y_test)
        dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)


        return dataloader_train, dataloader_val, dataloader_test, weight_dict
    def cal_loss_weight(self, y, beta=0.99999):

        labels_dict = dict(zip(range(self.num_task),[sum(y[:,i]) for i in range(self.num_task)]))
        # print(labels_dict)
        keys = labels_dict.keys()
        class_weight = dict()

        # Class-Balanced Loss Based on Effective Number of Samples
        for key in keys:
            if labels_dict[key] != 0:
                effective_num = 1.0 - beta**labels_dict[key]
                weights = (1.0 - beta) / effective_num
                class_weight[key] = weights
            else: 
                class_weight[key] = 0
        # print(class_weight)
        weights_sum = sum(class_weight.values())

        # normalizing weights
        for key in keys:
            class_weight[key] = class_weight[key] / weights_sum * 1

        return class_weight 
    
    def train_model(self, mRNA_ck_path = "/home/sxr280/DeepRBPLoc/Results/sequantial_fold0/checkpoints_33/epoch=73-step=25604.ckpt",
                    dataset = "/home/sxr280/DeepRBPLoc/new_data/lncRNA_all_data_seq_includingmouse_filtered.fasta",
                    batch_size = 64, RNA_type = "lncRNA", release_layer = 0):
        gin.parse_config_file('./Multihead_train_torch_sequential_modifymuchpossible.gin')

        ck_point = torch.load(mRNA_ck_path, map_location=torch.device(self.device))
        for key in list(ck_point["state_dict"].keys()):
            ck_point["state_dict"][key.replace("network.", "")] = ck_point["state_dict"].pop(key)
        dataloader_train, dataloader_val, dataloader_test, weight_dict = self.getdata(dataset = dataset, batch_size = batch_size)
        hyperparams_1 = {
                'fc_dim': 100,
                'weight_decay': 1e-5,
                'attention': True,####need to be changed
                'lr':0.001,
                'drop_flat':0.4,
                'drop_cnn': 0.3,
                'drop_input': 0.3,
                'hidden':256, #256
                'pooling_opt':True,
                'filter_length1':3,
                'activation':"gelu",
                'optimizer':"torch.optim.Adam",
                'release_layers': release_layer,#
                'prediction':False,
                'fc_layer' : True,
                'cnn_scaler': 1,
                'headnum': 3,
                'mode' : "full",
                'mfes' : False, ###
                'OHEM' : False, ###
                'loss_type': "fixed_weight",
                'class_weights': weight_dict,
                'gradient_clip': True,
                "add_neg" : False,
                'focal' : False,
                "nb_classes": self.num_task
                }

        model = myModel1(**hyperparams_1)
        net = model.network
        net = net.to(device = self.device)
        print("ck_point['state_dict']:",ck_point['state_dict'].keys())
        print("model['state_dict']:",net.state_dict().keys())

        #pop out the final layer, because of different classes
        ck_point["state_dict"].pop("FC_block.3.weight")
        ck_point["state_dict"].pop("FC_block.3.bias")
        ck_point["state_dict"].pop("fc2.weight")
        ck_point["state_dict"].pop("fc2.bias")
        ck_point["state_dict"].pop("fc3.weight")
        ck_point["state_dict"].pop("fc3.bias")


        #checking whether all the saved parameters are transferred into the model

        if self.load_weight:
            net.load_state_dict(ck_point['state_dict'], strict=False)

        model.network = net


        for i, (name, param) in enumerate(model.named_parameters()):
            print(name)
            # if name in ["network.FC_block.0.weight","network.FC_block.0.bias","network.FC_block.3.weight","network.FC_block.3.bias",]:
            # if name in ["network.FC_block.3.weight","network.FC_block.3.bias",]:

            if name in ["network.FC_block.0.weight","network.FC_block.0.bias","network.FC_block.3.weight","network.FC_block.3.bias","network.Attention1.W1","network.Attention1.W2","network.Attention2.W1","network.Attention2.W2"]:
                param = param.to(torch.float32)
                param.requires_grad = True
            else:
                param.requires_grad = False

        #model.network = net
        # if run<20:
        length = self.left+self.right
        summary(model, input_size = [(2,length),(2,int(length/8))], device = self.device)


        # %%
        OUTPATH = os.path.join(".",'Results/'+ "%s_fineruning" % RNA_type + '/')
        wandb_logger = WandbLogger(name = "release %s" % release_layer, project = "%s_finetune" % RNA_type, log_model = "all", save_dir = OUTPATH + "/checkpoints")
        # wandb_logger = WandbLogger(name = "scratch", project = "%s_finetune" % RNA_type, log_model = "all", save_dir = OUTPATH + "/checkpoints")

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name} is trainable")
            else:
                print(f"{name} is not trainable")
            
        trainer = Trainer(max_epochs = 500, gpus = 1, 
                logger = wandb_logger,
                log_every_n_steps = 1,
                callbacks = make_callback(OUTPATH, RNA_type, 20))

        trainer.fit(model, dataloader_train, dataloader_val)
        print("Saving the model not wrapped by pytorch-lighting")
        torch.save(model.network, OUTPATH + "/model%s_%s_%s.pth" % (RNA_type, release_layer, self.fold))

        #Doing the prediction
        print("----------Doing the evaluation----------")

        val = LncValidation(task = self.num_task)
        #doing the evaluation
        val.evaluation(light_model = model, left=self.left, right=self.right, RNA_type = RNA_type, testloader = dataloader_test)

 


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=3, help='the layers you want to release and do the fine-tuning')
    args = parser.parse_args()
    #the dataset must in the lncRNA directory
    # dataset = "/home/sxr280/DeepRBPLoc/new_data/lncRNA/lncRNA_all_data_seq_includingmouse_filtered.fasta"
    # dataset = "/home/sxr280/DeepRBPLoc/new_data/lncRNA/deeplncloc_dataset.fasta"
    dataset = "/home/sxr280/DeepRBPLoc/new_data/miRNA/miRNA_all_data_seq_filtered2.fasta"
    # /home/sxr280/DeepRBPLoc/new_data/deeplncloc_dataset.fasta
    #remove the file
    # import os
    # directory = "/home/sxr280/DeepRBPLoc/new_data/miRNA"
    # file_names = os.listdir(directory)
    # for file_name in file_names:
    #     file_path = os.path.join(directory, file_name)
    #     print(file_path)
    #     if file_name.startswith("Test") == True or file_name.startswith("Train") == True or file_name.startswith("Val") == True:
    #         print("removing", file_path)
    #         os.remove(file_path)
    load_weight = False
    for i in range(5):
        tune = lncTune(left = 50, right = 50, device = "cuda", num_task = 5, fold = i, load_weight = load_weight)
        # for layer in [1,3,7,20,30,45]:
        print(args)
        print("fine tune layers:", args.layer)
        tune.train_model(dataset = dataset, batch_size = 128, RNA_type = "miRNA", release_layer = args.layer)






