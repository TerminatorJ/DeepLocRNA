import sys
sys.path.insert(0, "../")
from Genedata import Gene_data
from preprocessing import *
import torch
from multihead_attention_model_torch import *
import gin
from sklearn.metrics import precision_recall_curve, average_precision_score,roc_auc_score,accuracy_score,matthews_corrcoef,f1_score,precision_score,recall_score
import time
from torchinfo import summary


class Evaluation:
    def __init__(self, ck_path = "/home/sxr280/DeepRBPLoc/Results/sequantial_fold0/checkpoints_33/epoch=73-step=25604.ckpt", task = 7):
        self.ck_path = ck_path
        self.task = task

    def get_label(self, label):
        
        labels = np.array(["Nucleus","Exsome","Cytosol","Cytoplasm","Ribosome","Membrane","Endoplasmic reticulum"])[:self.task]
        y = np.array(labels == label, dtype = "int")
        return list(y)

    def encode_data(self, dataset = "/home/sxr280/DeepRBPLoc/new_data/lncRNA_independent_set.txt",left=4000, right=4000):
        gene_data = Gene_data.load_sequence(dataset=dataset, left=left, right=right)
        id_label_seq_Dict = get_id_label_seq_Dict(gene_data)
        # print(id_label_seq_Dict)
        # print(id_label_seq_Dict)
        encoding_seq = OrderedDict([
            ('UNK', [0, 0, 0, 0]),
            ('A', [1, 0, 0, 0]),
            ('C', [0, 1, 0, 0]),
            ('G', [0, 0, 1, 0]),
            ('T', [0, 0, 0, 1]),
            ('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
        ])


        encoding_keys = list(encoding_seq.keys())
        y = np.array([self.get_label(id.split("|")[-1]) for id in id_label_seq_Dict.keys()])
        X_left = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][0]] for id in id_label_seq_Dict.keys()]
        X_right = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][1]] for id in id_label_seq_Dict.keys()]
                
            
        Xall = [np.concatenate([x,y],axis=-1) for x,y in zip(X_left,X_right)]
        # print([len(i) for i in Xall])
        X = np.array(pad_sequences(Xall,maxlen=left+right,dtype=np.int8, value=encoding_keys.index('UNK'),padding='post'))
        length = left + right
        mask_label=np.array([np.concatenate([np.ones(int(len(gene)/8)),np.zeros(int(length/8)-int(len(gene)/8))]) for gene in Xall],dtype='float32')
        return X, mask_label, y


    def evaluation(self, light_model = False, left=4000, right=4000, RNA_type = "lncRNA", testloader=None, RNA_index=None, device = "cuda"):
        
        
        if light_model == False:
            current_path = os.getcwd()
            config_file = os.path.join(current_path, "config.gin")
            gin.parse_config_file(config_file)

            ck_point = torch.load(self.ck_path, map_location=torch.device(device))
            for key in list(ck_point["state_dict"].keys()):
                ck_point["state_dict"][key.replace("network.", "")] = ck_point["state_dict"].pop(key)

            ini = myModel1()
            model = ini.network
            model = model.to(device = device)
            model.load_state_dict(ck_point['state_dict'], strict=True)
            model.eval()



        model = light_model
        model.eval()
        length = left + right
        try:
            summary(model, input_size = [(2,length),(2,int(length/8)), (2,length)], device = device)
        except:
            summary(model, input_size = [(2,length),(2,int(length/8))], device = device)
        if RNA_type == "lncRNA":
            dataloader_test = testloader
            roc_auc = self.get_metrics(model=model, dataloader_test=dataloader_test)
        #loading the parameters of mRNA model
        elif RNA_type in ["miRNA", "mRNA"]:
            dataloader_test = testloader
            roc_auc = self.get_metrics(model=model, dataloader_test=dataloader_test)
        elif RNA_type in ["allRNA"]:
            dataloader_test = testloader
            roc_auc = self.get_metrics_all(model=model, dataloader_test=dataloader_test, RNA_index = RNA_index)
        return roc_auc


    def get_metrics(self, model=None, dataloader_test=None):
        time1 = time.time()
        all_y_pred = []
        all_y_test = []
        roc_auc = dict()
        average_precision = dict()
        precision = dict()
        precision_sc = dict()
        recall = dict()
        recall_sc = dict()
        mcc_dict=dict()
        F1_score = dict()
        acc = dict()

        for i, batch in enumerate(dataloader_test):
            # print("doing:", i)
            try:
                X_test, X_mask, y_test = batch
                y_pred = model.forward(X_test, X_mask)
            except:
                X_test, X_tag, X_mask, y_test = batch
                y_pred = model.forward(X_test, X_mask, X_tag)

            
            y_pred = y_pred.detach().cpu().numpy()
            y_test = y_test.detach().cpu().numpy()
            all_y_pred.append(y_pred)
            all_y_test.append(y_test)

        y_test = np.concatenate(all_y_test, axis=0)
        y_pred = np.concatenate(all_y_pred, axis=0)
        for i in range(self.task):#calculate one by one
            try:
                average_precision[i+1] = average_precision_score(y_test[:, i], y_pred[:, i])
                roc_auc[i+1] = roc_auc_score(y_test[:,i], y_pred[:,i])
            except ValueError:
                average_precision[i+1] = np.nan
                roc_auc[i+1] = np.nan

            mcc_dict[i+1] = matthews_corrcoef(y_test[:,i],[1 if x>0.5 else 0 for x in y_pred[:,i]])
            F1_score[i+1] = f1_score(y_test[:,i],[1 if x>0.5 else 0 for x in y_pred[:,i]])
            precision[i+1], recall[i+1], _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
            acc[i+1] = accuracy_score(y_test[:,i],[1 if x>0.5 else 0 for x in y_pred[:,i]])
            precision_sc[i+1] = precision_score(y_test[:,i],[1 if x>0.5 else 0 for x in y_pred[:,i]])
            recall_sc[i+1] = recall_score(y_test[:,i],[1 if x>0.5 else 0 for x in y_pred[:,i]])
            print("y_test", str(i+1), y_test[:,i])
            print("y_pred", str(i+1), y_pred[:,i])



        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_pred.ravel())
        average_precision["micro"] = average_precision_score(y_test, y_pred,average="micro")
        roc_auc["micro"] = roc_auc_score(y_test,y_pred,average="micro")
        y_pred_bi = np.where(y_pred > 0.5, 1, 0)
        F1_score["micro"] = f1_score(y_test, y_pred_bi, average='micro')
        precision_sc["micro"] = precision_score(y_test, y_pred_bi, average='micro')
        recall_sc["micro"] = recall_score(y_test, y_pred_bi, average='micro')
        acc["micro"] = accuracy_score(y_test, y_pred_bi)
        time2 = time.time()

        print("auprc:", average_precision)
        print("roauc:", roc_auc)
        print("F1 score:", F1_score)
        print("acc score:", acc)
        print("precision score:", precision_sc)
        print("recall score:", recall_sc)
        print("mcc score:", mcc_dict)
        print("overall time elapse:", str(time2-time1))
        return roc_auc
    def get_metrics_all(self, model=None, dataloader_test=None, RNA_index = None):
        all_y_pred = []
        all_y_test = []

        for i, batch in enumerate(dataloader_test):
            try:
                X_test, X_mask, y_test = batch
                y_pred = model.forward(X_test, X_mask)
            except:
                X_test, X_tag, X_mask, y_test = batch
                y_pred = model.forward(X_test, X_mask, X_tag)
            
            y_pred = y_pred.detach().cpu().numpy()
            y_test = y_test.detach().cpu().numpy()
            all_y_pred.append(y_pred)
            all_y_test.append(y_test)
        y_test = np.concatenate(all_y_test, axis=0)
        y_pred = np.concatenate(all_y_pred, axis=0)

        self.print_metrics(y_test, y_pred, "allRNA")
        roc_auc_dict = {}
        RNA_types = ["lncRNA", "rRNA", "miRNA", "snRNA", "snoRNA", "mRNA"]
        for RNA in RNA_types:
            idx = np.where(np.isin(RNA_index, [RNA]))[0]
            if RNA == "lncRNA":
                idx = np.where(np.isin(RNA_index, ["lncRNA", "lincRNA"]))[0]
            
            try:
                roc_auc = self.print_metrics(y_test[idx], y_pred[idx], RNA)
                
            except:
                print("empty:", RNA)
                roc_auc = None
            roc_auc_dict[RNA] = roc_auc
        return roc_auc_dict

    def print_metrics(self, y_test, y_pred, RNA_type):
        time1 = time.time()
       
        roc_auc = dict()
        average_precision = dict()
        precision = dict()
        precision_sc = dict()
        recall = dict()
        recall_sc = dict()
        mcc_dict=dict()
        F1_score = dict()
        acc = dict()
        for i in range(self.task):#calculate one by one
            try:
                average_precision[i+1] = average_precision_score(y_test[:, i], y_pred[:, i])
                roc_auc[i+1] = roc_auc_score(y_test[:,i], y_pred[:,i])
            except ValueError:
                average_precision[i+1] = np.nan
                roc_auc[i+1] = np.nan
            mcc_dict[i+1] = matthews_corrcoef(y_test[:,i],[1 if x>0.5 else 0 for x in y_pred[:,i]])
            F1_score[i+1] = f1_score(y_test[:,i],[1 if x>0.5 else 0 for x in y_pred[:,i]])
            precision[i+1], recall[i+1], _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
            acc[i+1] = accuracy_score(y_test[:,i],[1 if x>0.5 else 0 for x in y_pred[:,i]])
            precision_sc[i+1] = precision_score(y_test[:,i],[1 if x>0.5 else 0 for x in y_pred[:,i]])
            recall_sc[i+1] = recall_score(y_test[:,i],[1 if x>0.5 else 0 for x in y_pred[:,i]])

        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_pred.ravel())
        average_precision["micro"] = average_precision_score(y_test, y_pred,average="micro")
        roc_auc["micro"] = roc_auc_score(y_test,y_pred,average="micro")
        y_pred_bi = np.where(y_pred > 0.5, 1, 0)
        F1_score["micro"] = f1_score(y_test, y_pred_bi, average='micro')
        precision_sc["micro"] = precision_score(y_test, y_pred_bi, average='micro')
        recall_sc["micro"] = recall_score(y_test, y_pred_bi, average='micro')
        acc["micro"] = accuracy_score(y_test, y_pred_bi)
        time2 = time.time()
        print("RNA type:", RNA_type)
        print("auprc:", average_precision)
        print("roauc:", roc_auc)
        print("F1 score:", F1_score)
        print("acc score:", acc)
        print("precision score:", precision_sc)
        print("recall score:", recall_sc)
        print("mcc score:", mcc_dict)
        print("overall time elapse:", str(time2-time1))
        return roc_auc

if __name__ == "__main__":
    eva = Evaluation()
    eva.evaluation()
