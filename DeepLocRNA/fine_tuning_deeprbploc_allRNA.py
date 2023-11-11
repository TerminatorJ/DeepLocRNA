import sys
sys.path.insert(0, "./data")
import torch
from preprocessing import *
from multihead_attention_model_torch import *
import gin
from torchinfo import summary
import argparse
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning import Trainer
import re
from pytorch_lightning.loggers import WandbLogger
from evaluation import Evaluation
torch.manual_seed(123)
np.random.seed(123)
current_path = os.getcwd()


class Finetune:
    def __init__(self, left = 4000, right = 4000, device = "cuda", num_task = 5, fold = 0, 
                DDP = False, load_data = False, gpu_num = 1, RNA_tag = False, species = "human", 
                flatten_tag = False, gradient_clip = False, lr = 0.001, loss_type = "BCE", 
                dim_attention = 80, fc_dim = 100, headnum = 3, jobnum = 3):
        self.left = left
        self.right = right
        self.device = device
        self.num_task = num_task
        self.fold = fold
        self.DDP = DDP
        self.gpu_num = gpu_num
        self.load_data = load_data
        self.RNA_tag = RNA_tag
        self.species = species
        self.flatten_tag = flatten_tag
        self.gradient_clip = gradient_clip
        self.lr = lr
        self.loss_type = loss_type
        self.dim_attention = dim_attention
        self.fc_dim = fc_dim
        self.headnum = headnum
        self.jobnum = jobnum
        self.save_path = None
    def get_binary(self, y):
        all_loc = ["Nucleus","Exosome","Cytosol","Cytoplasm","Ribosome","Membrane","Endoplasmic reticulum","Microvesicle", "Mitochondrion"]
        extra_idx = [all_loc.index(i) for i in ["Microvesicle","Exosome"]]
        intra_idx = [all_loc.index(j) for j in ["Nucleus","Cytosol","Cytoplasm","Ribosome","Membrane","Endoplasmic reticulum","Mitochondrion"]]
        extra_sum = y[:,extra_idx].sum(axis=1)
        intra_sum = y[:,intra_idx].sum(axis=1)
        new_y = np.column_stack((intra_sum, extra_sum))
        new_y = np.array(new_y, dtype = "bool")
        new_y = np.array(new_y, dtype = "int")
        return new_y
    def getdata(self, dataset=None, batch_size = 64, RNA_type = None):
        dataset_name = dataset.split("/")[-1].split(".")[0]
        print("running fold:", self.fold)
        current_path = os.getcwd()
        self.save_path = os.path.join(current_path, "data", RNA_type, dataset_name)


        # Check if the directory has been created
        if os.path.exists(self.save_path):
            print(f"Directory '{self.save_path}' already exist.")
        else:
            os.makedirs(self.save_path)
            print(f"Directory '{self.save_path}' has been created.")

        


        if not self.load_data:
            print("getting the data from regeneration", self.RNA_tag)
            Xtrain,Ytrain,Train_mask_label,Xtest,Ytest,Test_mask_label,Xval,Yval,Val_mask_label, encoding_keys, encoding_vectors = preprocess_data(left = self.left, right = self.right, dataset = dataset, padmod = "after", pooling_size=8,foldnum=5, pooling=True, RNA_type = RNA_type, RNA_tag = self.RNA_tag)
            if self.RNA_tag:
                for i in range(5):
                    
                    np.save(self.save_path + "/Train_fold%s_X_tag.npy" % str(i), Xtrain[i])
                    np.save(self.save_path + "/Test_fold%s_X_tag.npy" % str(i), Xtest[i])
                    np.save(self.save_path + "/Val_fold%s_X_tag.npy" % str(i), Xval[i])

                    np.save(self.save_path + "/Train_fold%s_y.npy" % str(i), Ytrain[i])
                    np.save(self.save_path + "/Test_fold%s_y.npy" % str(i), Ytest[i])
                    np.save(self.save_path + "/Val_fold%s_y.npy" % str(i), Yval[i])

                    np.save(self.save_path + "/Train_fold%s_mask.npy" % str(i), Train_mask_label[i])
                    np.save(self.save_path + "/Test_fold%s_mask.npy" % str(i), Test_mask_label[i])
                    np.save(self.save_path + "/Val_fold%s_mask.npy" % str(i), Val_mask_label[i])
                exit()

            else:
                for i in range(5):
                    print("saving the dir", self.save_path, os.path.join(self.save_path, "/Train_fold%s_X.npy" % str(i)))
                    np.save(self.save_path + "/Train_fold%s_X.npy" % str(i), Xtrain[i])
                    np.save(self.save_path + "/Test_fold%s_X.npy" % str(i), Xtest[i])
                    np.save(self.save_path + "/Val_fold%s_X.npy" % str(i), Xval[i])

                    np.save(self.save_path + "/Train_fold%s_y.npy" % str(i), Ytrain[i])
                    np.save(self.save_path + "/Test_fold%s_y.npy" % str(i), Ytest[i])
                    np.save(self.save_path + "/Val_fold%s_y.npy" % str(i), Yval[i])

                    np.save(self.save_path + "/Train_fold%s_mask.npy" % str(i), Train_mask_label[i])
                    np.save(self.save_path + "/Test_fold%s_mask.npy" % str(i), Test_mask_label[i])
                    np.save(self.save_path + "/Val_fold%s_mask.npy" % str(i), Val_mask_label[i])
                exit()
        else:
            assert os.path.exists(self.save_path + "/Train_fold%s_X.npy" % str(self.fold)), "please set load_data to False to generate data first"
            assert os.path.exists(self.save_path + "/Train_fold%s_X_tag.npy" % str(self.fold)), "don't forget to get the tag file, please set load_data to False and RNA_tag to True to generate the full dataset first"
            X_train_tag = np.load(self.save_path + "/Train_fold%s_X_tag.npy" % str(self.fold))
            X_train = np.load(self.save_path + "/Train_fold%s_X.npy" % str(self.fold))
            Y_train = np.load(self.save_path + "/Train_fold%s_y.npy" % str(self.fold))
            Train_mask = np.load(self.save_path + "/Train_fold%s_mask.npy" % str(self.fold))

            X_val_tag = np.load(self.save_path + "/Val_fold%s_X_tag.npy" % str(self.fold))
            X_val = np.load(self.save_path + "/Val_fold%s_X.npy" % str(self.fold))
            Y_val = np.load(self.save_path + "/Val_fold%s_y.npy" % str(self.fold))
            Val_mask = np.load(self.save_path + "/Val_fold%s_mask.npy" % str(self.fold))

            X_test_tag = np.load(self.save_path + "/Test_fold%s_X_tag.npy" % str(self.fold))
            X_test = np.load(self.save_path + "/Test_fold%s_X.npy" % str(self.fold))
            Y_test = np.load(self.save_path + "/Test_fold%s_y.npy" % str(self.fold))
            Test_mask = np.load(self.save_path + "/Test_fold%s_mask.npy" % str(self.fold))


            #make alignment with target size
            Y_train = Y_train[:,:self.num_task]
            Y_test = Y_test[:,:self.num_task]
            Y_val = Y_val[:,:self.num_task]

            print("X_train shape:", X_train.shape)
            print("Train_mask", Train_mask.shape)
            weight_dict = self.cal_loss_weight(Y_train, beta=0.99999)
            print("weight dict:", weight_dict)
            ratio = np.sum(Y_train, axis=0)/Y_train.shape[0]
            print("The ratio of each compartment:", ratio, np.sum(Y_train, axis=0))
            print("The test data of each compartment:", ratio, np.sum(Y_test, axis=0))
            #training data processes
            X_train = torch.from_numpy(X_train)#.to(self.device, torch.float)
            X_train_tag = torch.from_numpy(X_train_tag)#.to(self.device, torch.float)
            Y_train = torch.from_numpy(Y_train)#.to(self.device, torch.float)
            Train_mask = torch.from_numpy(Train_mask)#.to(self.device, torch.float)
            if self.RNA_tag:
                train_dataset = torch.utils.data.TensorDataset(X_train_tag, Train_mask, Y_train)
            elif self.RNA_tag == False:
                if not self.flatten_tag:
                    train_dataset = torch.utils.data.TensorDataset(X_train, Train_mask, Y_train)
                else:
                    train_dataset = torch.utils.data.TensorDataset(X_train, X_train_tag, Train_mask, Y_train)
            dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

            #validation data processes
            X_val = torch.from_numpy(X_val)#.to(self.device, torch.float)
            X_val_tag = torch.from_numpy(X_val_tag)#.to(self.device, torch.float)
            Y_val = torch.from_numpy(Y_val)#.to(self.device, torch.float)
            Val_mask = torch.from_numpy(Val_mask)#.to(self.device, torch.float)
            if self.RNA_tag:
                val_dataset = torch.utils.data.TensorDataset(X_val_tag, Val_mask, Y_val)
            elif self.RNA_tag == False:
                if not self.flatten_tag:
                    val_dataset = torch.utils.data.TensorDataset(X_val, Val_mask, Y_val)
                elif self.flatten_tag:
                    val_dataset = torch.utils.data.TensorDataset(X_val, X_val_tag, Val_mask, Y_val)
            dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

            #test data process
            X_test = torch.from_numpy(X_test)#.to(self.device, torch.float)
            X_test_tag = torch.from_numpy(X_test_tag)#.to(self.device, torch.float)
            Y_test = torch.from_numpy(Y_test)#.to(self.device, torch.float)
            Test_mask = torch.from_numpy(Test_mask)#.to(self.device, torch.float)
            if self.RNA_tag:
                test_dataset = torch.utils.data.TensorDataset(X_test_tag, Test_mask, Y_test)
            elif self.RNA_tag == False:
                if not self.flatten_tag:
                    test_dataset = torch.utils.data.TensorDataset(X_test, Test_mask, Y_test)
                else:
                    test_dataset = torch.utils.data.TensorDataset(X_test, X_test_tag, Test_mask, Y_test)
            
            dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
            



        return dataloader_train, dataloader_val, dataloader_test, weight_dict
    def GetDDPLoader(self, dataloader, batch_size):
        dataloader.shuffle = False
        # dataloader.batch_size = batch_size
        dataloader.pin_memory = True
        sampler = DistributedSampler(dataloader.dataset)
        dataloader.sampler = sampler
        return dataloader

    def cal_loss_weight(self, y, beta=0.99999, ER_enhance = False):

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
        #Mito performance is bad and not belong to mRNA, lower down its weights
        # class_weight[8] = class_weight[7]
        #lower down the performance of exosome
        # class_weight[1] = class_weight[1]*0.7
        if ER_enhance:
            #ER is more important in mRNA than mito and micro
            class_weight[6] = class_weight[6] * 2
            

        return class_weight 

    
    def train_model(self, mRNA_ck_path = "/home/sxr280/DeepRBPLoc/Results/sequantial_fold0/checkpoints_33/epoch=73-step=25604.ckpt",
                    dataset = "/home/sxr280/DeepRBPLoc/new_data/lncRNA_all_data_seq_includingmouse_filtered.fasta",
                    batch_size = 64, RNA_type = "lncRNA", release_layer = 0, evaluation = True):
        gin.parse_config_file('./config.gin')



        dataloader_train, dataloader_val, dataloader_test, weight_dict = self.getdata(dataset = dataset, batch_size = batch_size, RNA_type = RNA_type)
        dataset_name = dataset.split("/")[-1].split(".")[0]
        print("self.save_path", self.save_path)
        string = "".join(pickle.load(open(os.path.join(self.save_path, "Test5%s.pkl" % (self.fold)), "rb")))
        pattern = r"RNA_category:([^,\n]+)"
        RNA_index = np.array([i.split(">")[0] for i in re.findall(pattern, string)])
        print("RNA_index:", RNA_index)
        hyperparams_1 = {
                'fc_dim': self.fc_dim,
                'weight_decay': 1e-5,
                'attention': True,####need to be changed
                'lr': self.lr,
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
                'headnum': self.headnum,
                'mode' : "full",
                'mfes' : self.flatten_tag, ###
                'OHEM' : False, ###
                'loss_type': self.loss_type,
                'class_weights': weight_dict,
                'gradient_clip': self.gradient_clip,##
                "add_neg" : False,
                'focal' : False,
                "nb_classes": self.num_task,
                "RNA_type" : RNA_type,
                "dim_attention" : self.dim_attention,
                "species" : self.species
                }
       

        model = myModel1(**hyperparams_1)
        

        length = self.left+self.right
        try:
            summary(model, input_size = [(2,length),(2,int(length/8)),(2,length)], device = self.device)
        except:
            summary(model, input_size = [(2,length),(2,int(length/8))], device = self.device)

        OUTPATH = os.path.join(".",'Results/'+ "%s_finetuning" % RNA_type + '/')
        wandb_logger = WandbLogger(name = "release_layer: %s" % release_layer, project = "%s_finetune" % RNA_type, log_model = "all", save_dir = OUTPATH + "/checkpoints")

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name} is trainable")
            else:
                print(f"{name} is not trainable")

        #We set the machine running random value for each run
        run_value = self.jobnum
        print("the running value is:", run_value)

        if self.DDP:
            
            print("running ddp now")
            print("visible devices:", os.environ['CUDA_VISIBLE_DEVICES'])
            trainer = Trainer(accelerator="gpu", strategy='ddp', max_epochs = 500, devices = self.gpu_num, precision=32, num_nodes = 1, 
                logger = wandb_logger,
                log_every_n_steps = 1,
                callbacks = make_callback(OUTPATH, "%s_%s_%s_%s_%s_%s_%s" % (RNA_type, str(self.RNA_tag), release_layer, self.fold, self.species, self.flatten_tag, str(run_value)), 20))

        else:
            print("No DDP")
            trainer = Trainer(max_epochs = 500, gpus = self.gpu_num, 
                    logger = wandb_logger,
                    log_every_n_steps = 1,
                    callbacks = make_callback(OUTPATH, "%s_%s_%s_%s_%s_%s_%s" % (RNA_type, str(self.RNA_tag), release_layer, self.fold, self.species, self.flatten_tag, str(run_value)), 20))

        trainer.fit(model, dataloader_train, dataloader_val)
        print("Saving the model not wrapped by pytorch-lighting")


        #loading the best model after training
        try:
            print("loading the checkpoint:", 'checkpoints_%s_%s_%s_%s_%s_%s_%s_best.ckpt' % (RNA_type, str(self.RNA_tag), release_layer, self.fold, self.species, self.flatten_tag, str(run_value)))
            checkpoint = torch.load(OUTPATH + 'checkpoints_%s_%s_%s_%s_%s_%s_%s_best.ckpt' % (RNA_type, str(self.RNA_tag), release_layer, self.fold, self.species, self.flatten_tag, str(run_value)))
            model_state = checkpoint['state_dict']
            model.load_state_dict(model_state)

            loaded_param = model.state_dict()["network.fc1.weight"]
            trained_param = checkpoint['state_dict']["network.fc1.weight"]
            assert loaded_param==trained_param, "loaded parameters should be the same as trained parameters"

            torch.save(model.network, OUTPATH + "/model%s_%s_%s_%s_%s_%s_%s.pth" % (RNA_type, str(self.RNA_tag), release_layer, self.fold, self.species, self.flatten_tag, str(run_value)))


            #Doing the prediction
            print("----------Doing the evaluation----------")

            if evaluation:
                eva = Evaluation(task = self.num_task)
                #doing the evaluation
                eva.evaluation(light_model = model, left=self.left, right=self.right, RNA_type = RNA_type, testloader = dataloader_test, RNA_index = RNA_index)
        except:
            print("pass other task")


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=20, help='the layers you want to release and do the fine-tuning')
    parser.add_argument('--fold', type=int, default=3, help='the fold you want to release and do the fine-tuning')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of each epoch')
    parser.add_argument('--lr', type=float, default=0.005, help='The learning rate to train the model')

    parser.add_argument('--species', type=str, default="human", help='which species you are running')
    parser.add_argument("--dataset", type=str, default="/home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq.fasta")
    parser.add_argument("--RNA_tag", action = "store_true", default=False, help="add additional vector to represent RNA species")
    parser.add_argument("--flatten_tag", action = "store_true", default=False, help="add at the flatten layer")
    parser.add_argument("--load_data", action = "store_true", default=False, help="whether getting the data from saved files")
    parser.add_argument("--DDP", action = "store_true", default=False, help="Training the model in different gpus")
    parser.add_argument("--gradient_clip", action = "store_true", default=False, help="whether clip the gradient to prevent the gradient vanish and explosion")
    parser.add_argument('--loss_type', type=str, default="BCE", help='which weight to use of different classes')
    parser.add_argument('--dim_attention', type=int, default=50, help='the hidden dimension in the attention layer')
    parser.add_argument('--fc_dim', type=int, default=500, help="the width of the linear layer")
    parser.add_argument('--headnum', type=int, default=3, help="the number of attention heads we use in the attention layer")
    parser.add_argument('--jobnum', type=int, default=3, help="The number assigned to the job")
    parser.add_argument('--num_task', type=int, default=9, help="The number compartments that are used to build the model")
    parser.add_argument('--evaluation', action = "store_true", default=False, help="whether to conduct evaluation after training the model")



    parser.add_argument("--gpu_num", type=int)

    args = parser.parse_args()
    dataset = args.dataset

    import os
    os.environ["SIGTERM_TIMEOUT"] = "30"
 

    for i in range(5):
        tune = Finetune(left = 4000, right = 4000, device = "cuda", num_task = 9, fold = i, DDP = args.DDP, load_data = args.load_data, 
                        gpu_num = args.gpu_num, RNA_tag = args.RNA_tag, species = args.species, flatten_tag = args.flatten_tag, 
                        gradient_clip = args.gradient_clip, lr = args.lr, loss_type = args.loss_type, dim_attention = args.dim_attention,
                        fc_dim = args.fc_dim, headnum = args.headnum, jobnum = args.jobnum)
        # for layer in [1,3,7,20,30,45]:
        print(args)
        print("RNA_tag:", args.RNA_tag)
        print("fine tune layers:", args.layer)
        tune.train_model(dataset = dataset, batch_size = args.batch_size, RNA_type = "allRNA", release_layer = args.layer, evaluation = args.evaluation)


    # tune = lncTune(left = 4000, right = 4000, device = "cuda", num_task = 9, fold = args.fold, DDP = args.DDP, load_data = args.load_data, 
    #                     gpu_num = args.gpu_num, RNA_tag = args.RNA_tag, species = args.species, flatten_tag = args.flatten_tag, 
    #                     gradient_clip = args.gradient_clip, lr = args.lr, loss_type = args.loss_type, dim_attention = args.dim_attention,
    #                     fc_dim = args.fc_dim, headnum = args.headnum, jobnum = args.jobnum)
    # # for layer in [1,3,7,20,30,45]:
    # print(args)
    # print("RNA_tag:", args.RNA_tag)
    # print("fine tune layers:", args.layer)
    # tune.train_model(dataset = dataset, batch_size = args.batch_size, RNA_type = "allRNA", release_layer = args.layer)



# python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 20 --gpu_num 1 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_mouse_data_seq_deduplicated.fasta --RNA_tag --load_data --species mouse --DDP --batch_size 32
# python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 20 --gpu_num 1 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_mouse_data_seq_deduplicated.fasta --RNA_tag --load_data --species others --batch_size 32
# nohup python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 20 --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --DDP > allRNA_finetune_DDP4GPUnonredundantDM3Loc9tBCEhuman_5205.out &
# nohup python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 20 --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --DDP > allRNA_finetune_DDP4GPUnonredundantDM3Loc9tBCEnocliphuman_5205.out &

# nohup python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 20 --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --DDP --flatten_tag --gradient_clip > allRNA_finetune_DDP4GPUnonredundantDM3Loc9tBCEhuman_5205.out &

# nohup python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 20 --gpu_num 1 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --flatten_tag --gradient_clip > allRNA_finetune_nonDDP4GPUnonredundantDM3Loc9tBCEhuman_5205.out &

# nohup python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 20 --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 8 --flatten_tag --gradient_clip --DDP > allRNA_finetune_DDP4GPUnonredundantDM3Loc9tBCEhuman_5206.out &

# nohup python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 20 --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 8 --flatten_tag --gradient_clip --DDP --loss_type BCE --lr 0.0005 > allRNA_finetune_DDP4GPUnonredundantDM3Loc9tFlattenBCElr0.0005human_5206.out &
# nohup python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 20 --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 8 --flatten_tag --gradient_clip --DDP --loss_type BCE --lr 0.0001 > allRNA_finetune_DDP4GPUnonredundantDM3Loc9tFlattenBCElr0.0001human_5207.out &
#using different pre-trained infomation to get the good performance
# nohup python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 5 --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --flatten_tag --gradient_clip --DDP > allRNA_finetune_DDP4GPUnonredundantDM3Loc9tflattenBCEhumanrelease5.out &
# nohup python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 10 --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --flatten_tag --gradient_clip --DDP > allRNA_finetune_DDP4GPUnonredundantDM3Loc9tflattenBCEhumanrelease10.out &
# nohup python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 15 --gpu_num 2 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --flatten_tag --gradient_clip --DDP > allRNA_finetune_DDP4GPUnonredundantDM3Loc9tflattenBCEhumanrelease15.out &
# nohup python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 25 --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --flatten_tag --gradient_clip --DDP > allRNA_finetune_DDP4GPUnonredundantDM3Loc9tflattenBCEhumanrelease25.out &
# nohup python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 30 --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --flatten_tag --gradient_clip --DDP > allRNA_finetune_DDP4GPUnonredundantDM3Loc9tflattenBCEhumanrelease30.out &
# nohup python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 40 --gpu_num 2 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --flatten_tag --gradient_clip --DDP > allRNA_finetune_DDP4GPUnonredundantDM3Loc9tflattenBCEhumanrelease40.out &

#assign different weight to the classes to make ER balance with 7 labels, now it was suck, it needs to be improved.
#we use layer 20 to test
# nohup python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 20 --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --flatten_tag --gradient_clip --DDP --loss_type fixed_weight > allRNA_finetune_DDP4GPUnonredundantDM3Loc9tflattenBCEhumanrelease20weight8e7ERe.out &
# nohup python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 20 --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --flatten_tag --gradient_clip --DDP --loss_type fixed_weight > allRNA_finetune_DDP4GPUnonredundantDM3Loc9tflattenBCEhumanrelease20weight8e7EReE07.out &

#expand the model size
# nohup python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 20 --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --flatten_tag --gradient_clip --DDP --loss_type BCE --headnum 5 --fc_dim 150 --dim_attention 80 > allRNA_finetune_DDP4GPUnonredundantDM3Loc9tflattenBCEhumanrelease20515080.out &

# nohup python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 20 --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --flatten_tag --gradient_clip --DDP --loss_type BCE --headnum 8 --fc_dim 150 --dim_attention 80 > allRNA_finetune_DDP4GPUnonredundantDM3Loc9tflattenBCEhumanrelease20815080.out &

# nohup python /home/sxr280/DeepRBPLoc/fine_tuning_deeprbploc_allRNA.py --layer 20 --gpu_num 4 --dataset /home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta --load_data --species human --batch_size 16 --flatten_tag --gradient_clip --DDP --loss_type BCE > allRNA_finetune_DDP4GPUnonredundantDM3Loc9tflattenBCEhumanrelease2010150160.out &
