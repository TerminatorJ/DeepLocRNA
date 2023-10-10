# %% [markdown]
# Based on our pre-training model, we can do the fine-tuning

# %%
import sys
# sys.path.insert(0, "../")
from Multihead_train_torch_booster_modifymuchpossible import *
sys.path.insert(0, "./new_data")
import torch
from multihead_attention_model_torch_sequential_modifymuchpossible import *
# from Multihead_train_torch_booster_modifymuchpossible import evaluation

import gin
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score,roc_auc_score,accuracy_score,matthews_corrcoef,f1_score
import time
from torchinfo import summary
from cross_RNA_prediction import LncValidation
import argparse
# dataset = "/home/sxr280/DeepRBPLoc/new_data/lncRNA/lncRNA_all_data_seq_filtered.fasta"
# dataset = "/home/sxr280/DeepRBPLoc/new_data/lncRNA/lncRNA_all_data_seq_includingmouse.fasta"

class lncTune:
    def __init__(self, left = 4000, right = 4000, device = "cuda", num_task = 5, fold = 0, DDP = False, gpu_num = 1, save_data = False, weight = True):
        self.left = left
        self.right = right
        self.device = device
        self.num_task = num_task
        self.fold = fold
        self.DDP = DDP
        self.gpu_num = gpu_num
        self.save_data = save_data
        self.weight = weight
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
        dataset_name = dataset.split("/")[-1].split(".")[0]
        if self.save_data:
            Xtrain,Ytrain,Train_mask_label,Xtest,Ytest,Test_mask_label,Xval,Yval,Val_mask_label, encoding_keys, encoding_vectors = preprocess_data(left = self.left, right = self.right, dataset = dataset, padmod = "after", pooling_size=8,foldnum=5, pooling=True)

            np.save("/home/sxr280/DeepRBPLoc/new_data/allRNA/" + dataset_name + "/Train_fold%s_X_mRNA.npy" % str(self.fold), Xtrain[self.fold])
            np.save("/home/sxr280/DeepRBPLoc/new_data/allRNA/" + dataset_name + "/Test_fold%s_X_mRNA.npy" % str(self.fold), Xtest[self.fold])
            np.save("/home/sxr280/DeepRBPLoc/new_data/allRNA/" + dataset_name + "/Val_fold%s_X_mRNA.npy" % str(self.fold), Xval[self.fold])

            np.save("/home/sxr280/DeepRBPLoc/new_data/allRNA/" + dataset_name + "/Train_fold%s_y_mRNA.npy" % str(self.fold), Ytrain[self.fold][:,:self.num_task])
            np.save("/home/sxr280/DeepRBPLoc/new_data/allRNA/" + dataset_name + "/Test_fold%s_y_mRNA.npy" % str(self.fold), Ytest[self.fold][:,:self.num_task])
            np.save("/home/sxr280/DeepRBPLoc/new_data/allRNA/" + dataset_name + "/Val_fold%s_y_mRNA.npy" % str(self.fold), Yval[self.fold][:,:self.num_task])

            np.save("/home/sxr280/DeepRBPLoc/new_data/allRNA/" + dataset_name + "/Train_fold%s_mask_mRNA.npy" % str(self.fold), Train_mask_label[self.fold])
            np.save("/home/sxr280/DeepRBPLoc/new_data/allRNA/" + dataset_name + "/Test_fold%s_mask_mRNA.npy" % str(self.fold), Test_mask_label[self.fold])
            np.save("/home/sxr280/DeepRBPLoc/new_data/allRNA/" + dataset_name + "/Val_fold%s_mask_mRNA.npy" % str(self.fold), Val_mask_label[self.fold])
        else:
            print("Loading the data:", "/home/sxr280/DeepRBPLoc/new_data/allRNA/" + dataset_name + "/Train_fold%s_X_mRNA.npy" % str(self.fold))
            X_train = np.load("/home/sxr280/DeepRBPLoc/new_data/allRNA/" + dataset_name + "/Train_fold%s_X_mRNA.npy" % str(self.fold))
            X_test = np.load("/home/sxr280/DeepRBPLoc/new_data/allRNA/" + dataset_name + "/Test_fold%s_X_mRNA.npy" % str(self.fold))
            X_val = np.load("/home/sxr280/DeepRBPLoc/new_data/allRNA/" + dataset_name + "/Val_fold%s_X_mRNA.npy" % str(self.fold))

            Y_train = np.load("/home/sxr280/DeepRBPLoc/new_data/allRNA/" + dataset_name + "/Train_fold%s_y_mRNA.npy" % str(self.fold))[:,:self.num_task]
            Y_test = np.load("/home/sxr280/DeepRBPLoc/new_data/allRNA/" + dataset_name + "/Test_fold%s_y_mRNA.npy" % str(self.fold))[:,:self.num_task]
            Y_val = np.load("/home/sxr280/DeepRBPLoc/new_data/allRNA/" + dataset_name + "/Val_fold%s_y_mRNA.npy" % str(self.fold))[:,:self.num_task]

            Train_mask = np.load("/home/sxr280/DeepRBPLoc/new_data/allRNA/" + dataset_name + "/Train_fold%s_mask_mRNA.npy" % str(self.fold))
            Test_mask = np.load("/home/sxr280/DeepRBPLoc/new_data/allRNA/" + dataset_name + "/Test_fold%s_mask_mRNA.npy" % str(self.fold))
            Val_mask = np.load("/home/sxr280/DeepRBPLoc/new_data/allRNA/" + dataset_name + "/Val_fold%s_mask_mRNA.npy" % str(self.fold))


        

        print("running fold:", self.fold)
        print("X_train shape:", X_train.shape)
        print("Train_mask", Train_mask.shape)
        print("batch size:", batch_size)
        print("batches in each epoch:", X_train.shape[0]/batch_size)
        if self.weight:
            
        # weight_dict = {0: 0.071193216473815235, 1: 0.02807763758305944, 2: 0.17593567881871017, 3: 0.09767276889753683, 4: 0.18077738075144945, 5: 0.19682227759651573, 6: 0.28952103987891316}
            weight_dict = dict(zip([i for i in range(self.num_task)], [j/sum([1,1,7,1,3,5,8]) for j in [1,1,7,1,3,5,8]]))
        else:
            weight_dict = self.cal_loss_weight(Y_train, beta=0.99999)
        print("The weight of each compartment are: ", weight_dict)
        ratio = np.sum(Y_train, axis=0)/Y_train.shape[0]
        print("The ratio of each compartment:", ratio, np.sum(Y_train, axis=0))
        print("The test data of each compartment:", ratio, np.sum(Y_test, axis=0))
        #training data processes
        X_train = torch.from_numpy(X_train)
        Y_train = torch.from_numpy(Y_train)
        Train_mask = torch.from_numpy(Train_mask)
        train_dataset = torch.utils.data.TensorDataset(X_train, Train_mask, Y_train)
        dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

        #validation data processes
        X_val = torch.from_numpy(X_val)
        Y_val = torch.from_numpy(Y_val)
        Val_mask = torch.from_numpy(Val_mask)
        val_dataset = torch.utils.data.TensorDataset(X_val, Val_mask, Y_val)
        dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
        
        #test data process
        X_test = torch.from_numpy(X_test)
        Y_test = torch.from_numpy(Y_test)
        Test_mask = torch.from_numpy(Test_mask)
        test_dataset = torch.utils.data.TensorDataset(X_test, Test_mask, Y_test)
        dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size = 8, shuffle = False)


        #saving the dataset
        # /home/sxr280/DeepRBPLoc/testdata/Test_fold0_mask.npy



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
                'loss_type': "fixed_weight",##
                'class_weights': weight_dict,
                'gradient_clip': True,##
                "add_neg" : False,
                'focal' : False,
                "nb_classes": self.num_task,
                "dataset" : dataset,
                "RNA_type" : RNA_type
                }
       

        model = myModel1(**hyperparams_1)
        

        #checking whether all the saved parameters are transferred into the model


       
        #model.network = net
        # if run<20:
        length = self.left+self.right
        summary(model, input_size = [(2,length),(2,int(length/8))], device = self.device)


        # %%
        OUTPATH = os.path.join(".",'Results/'+ "%s_fineruning" % RNA_type + '/')
        # wandb_logger = WandbLogger(name = "release %s" % release_layer, project = "%s_finetune" % RNA_type, log_model = "all", save_dir = OUTPATH + "/checkpoints")
        wandb_logger = WandbLogger(name = "release_layer: %s" % release_layer, project = "%s_finetune" % RNA_type, log_model = "all", save_dir = OUTPATH + "/checkpoints")

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name} is trainable")
            else:
                print(f"{name} is not trainable")
        if self.DDP:
            trainer = Trainer(accelerator="cuda", strategy='ddp', max_epochs = 500, devices = self.gpu_num, precision=32, num_nodes = 1, 
                logger = wandb_logger,
                log_every_n_steps = 1,
                callbacks = make_callback(OUTPATH, RNA_type, 20))
            # trainer = Trainer(accelerator="cuda", strategy='ddp', max_epochs = 500, devices = [0,1], precision=32, num_nodes = 1, 
            #     logger = wandb_logger,
            #     log_every_n_steps = 1,
            #     callbacks = make_callback(OUTPATH, RNA_type, 20))
        else:
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
    parser.add_argument('--fold', type=int, default=3, help='the fold you want to release and do the fine-tuning')
    # parser.add_argument("--local_rank", type=int)
    parser.add_argument("--DDP", action = "store_true", default=False)
    parser.add_argument("--dataset", type=str, default="/home/sxr280/DeepRBPLoc/testdata/modified_multilabel_seq_nonredundent.fasta")
    parser.add_argument("--weight", action = "store_true", default=False, help="add additional weights to different compartments")
    parser.add_argument("--gpu_num", type=int)
    args = parser.parse_args()
    #the dataset must in the lncRNA directory
    # dataset = "/home/sxr280/DeepRBPLoc/new_data/lncRNA/lncRNA_all_data_seq_includingmouse_filtered.fasta"
    # dataset = "/home/sxr280/DeepRBPLoc/new_data/lncRNA/deeplncloc_dataset.fasta"
    # dataset = "/home/sxr280/DeepRBPLoc/testdata/modified_multilabel_seq_nonredundent.fasta"
    dataset = args.dataset
    # /home/sxr280/DeepRBPLoc/new_data/deeplncloc_dataset.fasta
    #remove the file
    import os
    # directory = "/home/sxr280/DeepRBPLoc/testdata"
    # file_names = os.listdir(directory)
    # for file_name in file_names:
    #     file_path = os.path.join(directory, file_name)
    #     print(file_path)
    #     if file_name.startswith("Test5") == True or file_name.startswith("Train5") == True or file_name.startswith("Val5") == True:
    #         print("removing", file_path)
    #         os.remove(file_path)

    for i in range(5):
        tune = lncTune(left = 4000, right = 4000, device = "cuda", num_task = 7, fold = i, DDP = args.DDP, gpu_num = args.gpu_num, save_data = False, weight = args.weight)
        # for layer in [1,3,7,20,30,45]:
        print(args)
        print("fine tune layers:", args.layer)
        print("DDP:", args.DDP)
        print("weight:", args.weight)
        tune.train_model(dataset = dataset, batch_size = 32, RNA_type = "mRNA", release_layer = args.layer)




    #i want to fix the crash issue
    # print(args)
    # print("fine tune layers:", args.layer)
    # print("fold:", args.fold)
    # tune = lncTune(left = 4000, right = 4000, device = "cuda", num_task = 7, fold = args.fold)
    # # for layer in [1,3,7,20,30,45]:
    # tune.train_model(dataset = dataset, batch_size = 32, RNA_type = "mRNA", release_layer = args.layer)







