from collections import OrderedDict
import random
import torch
import numpy as np
from DeepLocRNA.DeepLocRNA.multihead_attention_model_torch import *
from sklearn.metrics import average_precision_score,roc_auc_score,accuracy_score,matthews_corrcoef,f1_score,precision_score,recall_score
import os
import wandb
from preprocessing import *
from utils import *

device = "cuda"

def run_parnet(data , device, batch_size_parnet):
    encoding_seq = OrderedDict([
            ('UNK', [0, 0, 0, 0]),
            ('A', [1, 0, 0, 0]),
            ('C', [0, 1, 0, 0]),
            ('G', [0, 0, 1, 0]),
            ('T', [0, 0, 0, 1]),
            ('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
        ])
    # device = "cpu"
    embedding_vec = np.array(list(encoding_seq.values()))

    parnet_model = torch.load("/home/sxr280/DeepRBPLoc/parnet_model/network.PanRBPNet.2023-03-13.ckpt")
    embedding_layer = nn.Embedding(num_embeddings=len(embedding_vec),embedding_dim=len(embedding_vec[0]),_weight=torch.tensor(embedding_vec))
    embedding_layer = embedding_layer.to(device=device)
    parnet_model = parnet_model.to(device=device)
    for param in parnet_model.parameters():
        param.requires_grad = False

    parnet_model.eval()
    parnet_out = np.zeros((data.shape[0], 1, 8000))
    # print("parnet_out ini:", parnet_out.shape)
    totalbatchs = int(np.ceil(float(data.shape[0]/batch_size_parnet)))
    
    for batch in range(totalbatchs):
        x = data[batch*batch_size_parnet: min((batch+1)*batch_size_parnet, data.shape[0])]
        # if device == "cpu":
        x = x.to(device)
        x = x.long()
        embedding_output = embedding_layer(x)
        embedding_output = embedding_output.transpose(1,2)
        
        # print("embedding_output:",embedding_output.shape)
        out = parnet_model.forward(embedding_output)
        # print("parnet output:", out, out.shape)
        out = out.detach().cpu().numpy()
        # print("batch:", batch)
        # print("batch*batch_size_parnet", batch*batch_size_parnet)
        # print("batch*(batch_size_parnet + 1)", batch*(batch_size_parnet + 1))
        # print("data.shape[0]", data.shape[0])
        # print("min(batch*(batch_size_parnet + 1), data.shape[0])", min(batch*(batch_size_parnet + 1), data.shape[0]))
        parnet_out[batch*batch_size_parnet: min((batch+1)*batch_size_parnet, data.shape[0])] = out
    return parnet_out


def evaluation(fold, batch_size, model, mfes):
    model = model.to(device)
    model.eval()
    roc_auc = dict()
    average_precision = dict()
    mcc_dict=dict()
    F1_score = dict()
    precision_sc = dict()
    recall_sc = dict()
    acc = dict()

    if mfes:
        y_test = np.load("./testdata/Test_fold%s_y.npy" % fold)
        X_test = np.load("./testdata/Test_fold%s_X.npy" % fold)
        x_mask = np.load("./testdata/Test_fold%s_mask.npy" % fold)
        x_mfes = np.load("./testdata/Test_fold%s_mfes_X.npy" % fold)

        y_test = torch.from_numpy(y_test)#.to(device, torch.float)
        X_test = torch.from_numpy(X_test)#.to(device, torch.float)
        X_mask = torch.from_numpy(x_mask)#.to(device, torch.float)
        X_mfes = torch.from_numpy(x_mfes)

        test_dataset = torch.utils.data.TensorDataset(X_test, X_mfes, X_mask, y_test)
        dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

        all_y_pred = []
        all_y_test = []

    
        for i, batch in enumerate(dataloader_test):
            print("doing evaluation:", i)
            X_test, X_mfes, X_mask, y_test = batch
            print("device check out")
            print(X_test.device)
            print(X_mask.device)
            print(model.device)
            y_pred = model.forward(X_test, X_mask, X_mfes)
            y_pred = y_pred.detach().cpu().numpy()
            y_test = y_test.detach().cpu().numpy()
            all_y_pred.append(y_pred)
            all_y_test.append(y_test)

    else:
        y_test = np.load("./testdata/Test_fold%s_y.npy" % fold)
        X_test = np.load("./testdata/Test_fold%s_X.npy" % fold)
        x_mask = np.load("./testdata/Test_fold%s_mask.npy" % fold)

        y_test = torch.from_numpy(y_test)#.to(device, torch.float)
        X_test = torch.from_numpy(X_test)#.to(device, torch.float)
        X_mask = torch.from_numpy(x_mask)#.to(device, torch.float)

        test_dataset = torch.utils.data.TensorDataset(X_test, X_mask, y_test)
        dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

        all_y_pred = []
        all_y_test = []

    
        for i, batch in enumerate(dataloader_test):
            print("doing evaluation:", i)
            X_test, X_mask, y_test = batch
            print("device check out")
            print(X_test.device)
            print(X_mask.device)
            print(model.device)
            y_pred = model.forward(X_test, X_mask)
            y_pred = y_pred.detach().cpu().numpy()
            y_test = y_test.detach().cpu().numpy()
            all_y_pred.append(y_pred)
            all_y_test.append(y_test)

    

    y_test = np.concatenate(all_y_test, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    print("y_test shape", y_test.shape)
    print("y_test shape", y_pred.shape)
    for i in range(7):#calculate one by one
        average_precision.setdefault(fold,{})[i+1] = average_precision_score(y_test[:, i], y_pred[:, i])
        roc_auc.setdefault(fold,{})[i+1] = roc_auc_score(y_test[:,i], y_pred[:,i])
        mcc_dict.setdefault(fold,{})[i+1] = matthews_corrcoef(y_test[:,i],[1 if x>0.5 else 0 for x in y_pred[:,i]])
        F1_score.setdefault(fold,{})[i+1] = f1_score(y_test[:,i],[1 if x>0.5 else 0 for x in y_pred[:,i]])
        acc[i+1] = accuracy_score(y_test[:,i],[1 if x>0.5 else 0 for x in y_pred[:,i]])
        precision_sc[i+1] = precision_score(y_test[:,i],[1 if x>0.5 else 0 for x in y_pred[:,i]])
        recall_sc[i+1] = recall_score(y_test[:,i],[1 if x>0.5 else 0 for x in y_pred[:,i]])

    average_precision.setdefault(fold,{})["micro"] = average_precision_score(y_test, y_pred,average="micro")
    roc_auc.setdefault(fold,{})["micro"] = roc_auc_score(y_test,y_pred,average="micro")
    y_pred_bi = np.where(y_pred > 0.5, 1, 0)
    F1_score.setdefault(fold,{})["micro"] = f1_score(y_test, y_pred_bi, average='micro')
    precision_sc["micro"] = precision_score(y_test, y_pred_bi, average='micro')
    recall_sc["micro"] = recall_score(y_test, y_pred_bi, average='micro')
    acc["micro"] = accuracy_score(y_test, y_pred_bi)
    # print("run", run)
    print("auprc:", average_precision)
    print("roauc:", roc_auc)
    print("F1 score:", F1_score)
    print("mcc score:", mcc_dict)
    print("acc score:", acc)
    print("precision score:", precision_sc)
    print("recall score:", recall_sc)




encoding_seq = OrderedDict([
    ('UNK', [0, 0, 0, 0]),
    ('A', [1, 0, 0, 0]),
    ('C', [0, 1, 0, 0]),
    ('G', [0, 0, 1, 0]),
    ('T', [0, 0, 0, 1]),
    ('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
])
basedir = os.getcwd()

encoding_fold = {'(': [1, 0, 0, 0],
                ')': [0, 1, 0, 0],
                '.': [0, 0, 1, 0],
                'N': [0, 0, 0, 1]}

seq_encoding_keys = list(encoding_seq.keys())
# global seq_encoding_keys
fold_encoding_keys = list(encoding_fold.keys())
seq_encoding_vectors = np.array(list(encoding_seq.values()))

def run_model(dataset='/home/sxr280/DeepRBPLoc/new_data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2.fasta',
              pooling_size = 8,
              pooling = True,
              left = 4000, 
              right = 4000, 
              padmod = "after", 
              foldnum = 5, 
              gpu_num = 1,
              run = 33,
              max_epochs = 500,
              message = "sequantial_fold0",
              mfes = False,
              add_neg = False):
    embedding_vec = seq_encoding_vectors
    OUTPATH = os.path.join(basedir,'Results/'+message + '/')
    if not os.path.exists(OUTPATH):
        os.makedirs(OUTPATH)
    datasetfolder=os.path.dirname(dataset)
    wandb.login(key="57f4851d7943ea1dec3b10273876045d051b40f1")
    api = wandb.Api(timeout=19)
    Xtrain,Ytrain,Train_mask_label,Xtest,Ytest,Test_mask_label,Xval,Yval,Val_mask_label, encoding_keys, encoding_vectors = preprocess_data(left = left, right = right, dataset = dataset, padmod = padmod, pooling_size=pooling_size,foldnum=foldnum, pooling=pooling)


    # print("saving the test dataset:...")
    i = int(message[-1])
    fold = i

    #loading the saved RNA fold structure
    print("loading the saved RNA structure")
    
                  
    # #loading the datasets that are preprocessed
    Xtrain = torch.from_numpy(Xtrain[i])#.to(device, torch.int)
    if mfes:
        Xtrain_mfes = np.load("/home/sxr280/DeepRBPLoc/testdata/Train_fold%s_mfes_X.npy" % fold)
        # print("loaded Xtrain_mfes shape:", Xtrain_mfes.shape)
        Xtrain_mfes = torch.from_numpy(Xtrain_mfes)#.to(device, torch.int)
    Ytrain = torch.from_numpy(Ytrain[i])#.to(device, torch.float)
    weight_dict = cal_loss_weight(Ytrain, beta=0.99999)
    Train_mask_label = torch.from_numpy(Train_mask_label[i])#.to(device, torch.int)

    Xval = torch.from_numpy(Xval[i])#.to(device, torch.int)
    if mfes:
        Xval_mfes = np.load("/home/sxr280/DeepRBPLoc/testdata/Val_fold%s_mfes_X.npy" % fold)
        Xval_mfes = torch.from_numpy(Xval_mfes)#.to(device, torch.int)
    Yval = torch.from_numpy(Yval[i])#.to(device, torch.float)
    Val_mask_label = torch.from_numpy(Val_mask_label[i])#.to(device, torch.int)


    Xtest = torch.from_numpy(Xtest[i])#.to(device, torch.int)
    if mfes:
        Xtest_mfes = np.load("/home/sxr280/DeepRBPLoc/testdata/Test_fold%s_mfes_X.npy" % fold)
        Xtest_mfes = torch.from_numpy(Xtest_mfes[i])#.to(device, torch.int)
    Ytest = torch.from_numpy(Ytest[i])#.to(device, torch.float)
    Test_mask_label = torch.from_numpy(Test_mask_label[i])#.to(device, torch.int)

    if mfes:
        # print("Xtrain, Xtrain_mfes", Xtrain.shape, Xtrain_mfes.shape)
        train_dataset = torch.utils.data.TensorDataset(Xtrain, Xtrain_mfes, Train_mask_label, Ytrain)
        val_dataset = torch.utils.data.TensorDataset(Xval, Xval_mfes, Val_mask_label, Yval)
    else:
        train_dataset = torch.utils.data.TensorDataset(Xtrain, Train_mask_label, Ytrain)
        val_dataset = torch.utils.data.TensorDataset(Xval, Val_mask_label, Yval)
    # train_dataset = torch.utils.data.TensorDataset(Xtrain, Train_mask_label, Ytrain)
    # val_dataset = torch.utils.data.TensorDataset(Xval, Val_mask_label, Yval)

    #parameters that you can do the gridsearch
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
        'release_layers': 20,
        'prediction':False,
        'fc_layer' : True,
        'cnn_scaler': 1,
        'headnum': 3,
        'mode' : "full",
        'mfes' : mfes, ###
        'OHEM' : False, ###
        'loss_type': "BCE",
        'class_weights': weight_dict,
        'gradient_clip': True,
        "add_neg" : add_neg,
        'focal' : False,
        "dataset" : dataset,
        "RNA_type" : "mRNA",
        "nb_classes": 9
        }
    
    hyperparams_2 = {
        'filters_length1': 5,
        'filters_length2': 20,
        'filters_length3': 60,
        'headnum': 3,
        'nb_filters': 64,
        'hidden': 32,
        'dim_attention': 80,
        'mode' : "feature"
        }
    #fusion: use DM3Loc(get after attention layer) and parnet(get 256 layer) structure.
    #
    hyperparams_3 = {
        'hidden': 544,
        'fc_dim': 100,
        'drop_flat': 0.4,
        'batch_size': 16,#
        'patience':20,
        'mode':"fusion"
        }
    
    hyperparams = {
        "param_1": hyperparams_1,
        "param_2": hyperparams_2,
        "param_3": hyperparams_3,
        }

    wandb_logger = WandbLogger(name = str(hyperparams_1), project = "5_folds_fusion_2_paper", log_model = "all", save_dir = OUTPATH + "/checkpoints_%s" % run)
    # for index, params  in enumerate(ParameterGrid(hyperparams)):
    index = 0
    print("running the fold: ", fold)
    print("Doing:", hyperparams)
    # print("this is the leanable weight test")

    params_1 = hyperparams["param_1"]
    params_2 = hyperparams["param_2"]
    params_3 = hyperparams["param_3"]

    batch_size_params = {'batch_size': params_3.pop('batch_size')}
    patience_params = {'patience': params_3.pop('patience')}


    params_3["batch_size"] = batch_size_params['batch_size']
    params_3["patience"] = patience_params['patience']
    # model = model.to(device = device)

    model = myModel1(**params_1)
    if mfes:  
        if pooling:
            summary(model, input_size = [(2,8000),(2,1000),(2,8000)], device = device)
        else:
            summary(model, input_size = [(2,8000),(2,8000),(2,8000)], device = device)
    else:
        if pooling:
            summary(model, input_size = [(2,8000),(2,1000)], device = device)
        else:
            summary(model, input_size = [(2,8000),(2,8000)], device = device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} is trainable")
        else:
            print(f"{name} is not trainable")
    # dataloader_train = torch.utils.data.DataLoader(train_dataset_mfes, batch_size = batch_size_params['batch_size'], shuffle = True)
    # dataloader_val = torch.utils.data.DataLoader(val_dataset_mfes, batch_size = batch_size_params['batch_size'], shuffle = True)

    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size_params['batch_size'], shuffle = True)
    dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size_params['batch_size'], shuffle = True)

    # with wandb.init(reinit=True):
    
    trainer = Trainer(max_epochs = max_epochs, gpus = gpu_num, 
            logger = wandb_logger,
            log_every_n_steps = 1,
            callbacks = make_callback(OUTPATH, str(run), patience_params['patience']))

    trainer.fit(model, dataloader_train, dataloader_val)
    print("Saving the model not wrapped by pytorch-lighting")
    torch.save(model.network, OUTPATH + "/model%s_%s.pth" % (run, index))
    
    #Doing the prediction
    print("----------Doing the evaluation----------")
    
    evaluation(fold, 8, model, mfes)
 


if __name__ == "__main__":
    gin.parse_config_file('./config.gin')
    run_model()    