from collections import OrderedDict
import re
import numpy as np
import random
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def get_id_label_seq_Dict(gene_data):
    id_label_seq_Dict = OrderedDict()
    for gene in gene_data:
         label = gene.label
        #  print("label:", label)
         gene_id = gene.id.strip()
         # print("gene id:", gene_id)
         id_label_seq_Dict[gene_id] = {}
         id_label_seq_Dict[gene_id][label]= (gene.seqleft,gene.seqright)
    
    return id_label_seq_Dict

def get_label_id_Dict(id_label_seq_Dict):
    label_id_Dict = OrderedDict()
    for eachkey in id_label_seq_Dict.keys():
        label = list(id_label_seq_Dict[eachkey].keys())[0]
        label_id_Dict.setdefault(label,set()).add(eachkey)
    
    return label_id_Dict

def label_dist(dist):
    label = []
    for x in dist:
        try:
            label.append(int(x))
        except:
            continue

    return label

# def get_fold(seq):
#     fc = RNA.fold_compound(seq)
#     # compute MFE and MFE structure
#     (mfe_struct, mfe) = fc.mfe()
#     return mfe_struct

def get_new_seq(input_types, Xall, encoding_keys, left, right):
    Xall2 = []
    for seq in Xall:
        # pattern = r'RNA_category:([^,\n]+)'
        # RNA_types = re.findall(pattern, id_tag)
        # RNA_tag = encoding_keys.index(RNA_types[0])
        RNA_tag = encoding_keys.index(input_types)
        seq2 = np.insert(seq, 0, RNA_tag)
        if len(seq) < (left+right):
            Xall2.append(seq2)
        else:
            Xall2.append(seq2[:left+right])
    return Xall2

def get_new_seq_train(ids, Xall, encoding_keys, left, right):
    Xall2 = []
    for seq, id_tag in zip(Xall, ids):
        pattern = r'RNA_category:([^,\n]+)'
        RNA_types = re.findall(pattern, id_tag)
        RNA_tag = encoding_keys.index(RNA_types[0])
        # RNA_tag = encoding_keys.index(input_types)
        seq2 = np.insert(seq, 0, RNA_tag)
        if len(seq) < (left+right):
            Xall2.append(seq2)
        else:
            Xall2.append(seq2[:left+right])
    return Xall2

def cal_loss_weight(y, beta=0.99999):

    num_task = 7
    labels_dict = dict(zip(range(num_task),[sum(y[:,i]) for i in range(num_task)]))
    keys = labels_dict.keys()
    class_weight = dict()

    # Class-Balanced Loss Based on Effective Number of Samples
    for key in keys:
        effective_num = 1.0 - beta**labels_dict[key]
        weights = (1.0 - beta) / effective_num
        class_weight[key] = weights

    weights_sum = sum(class_weight.values())

    # normalizing weights
    for key in keys:
        class_weight[key] = class_weight[key] / weights_sum * 1

    return class_weight 

def neg_gen(seed, left, right, type):
    random.seed(seed)
    elements = [0, 1, 2, 3]
    seq_length = left+right
    if type == "seq":
        sequence = [random.choice(elements) for _ in range(seq_length)]
        return sequence
    elif type == "mask":
        mask = np.ones(int(seq_length/8))
        begin = random.randint(0, 4000)
        mask[-begin:] = 0
        return mask
    elif type == "y":
        y = np.zeros(7)
        return y
    
    return sequence


class PredictionCallback(pl.Callback):
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def on_batch_end(self, trainer, pl_module):
        x, mask, y = next(iter(self.dataloader))
        x = x.to(pl_module.device)
        y_pred = pl_module(x, mask)
        print('Batch', trainer.global_step, 'training predictions:', y_pred)
def make_callback(output_path, msg, patience):
    """
    save the parameters we trained during each epoch.
    Params:
    -------
    output_path: str,
        the prefix of the path to save the checkpoints.
    freeze: bool,
        whether to freeze the first section of the model. 
    """
    # files = os.listdir(output_path)
    # for f in files:
    #     if f.startswith("checkpoints_%s_best" % str(msg)):
    #         file_path = os.path.join(output_path, f)
    #         os.remove(file_path)
    #         print(f"Deleted file: {file_path}")
    callbacks = [
        ModelCheckpoint(dirpath = output_path, filename = "checkpoints_%s_best" % str(msg), save_top_k = 1, verbose = True, mode = "min", monitor = "val_loss"),
        EarlyStopping(monitor = "val_loss", min_delta = 0.00, patience = patience, verbose = True, mode = "min"),
        ]
    return callbacks

def _make_loggers(output_path, msg):
    loggers = [
        pl_loggers.TensorBoardLogger(output_path + '/logger', name=str(msg), version='', log_graph=True),
    ]
    return loggers


def neg_gen(seed, left, right, type):
    random.seed(seed)
    elements = [0, 1, 2, 3]
    seq_length = left+right
    if type == "seq":
        sequence = [random.choice(elements) for _ in range(seq_length)]
        return sequence
    elif type == "mask":
        mask = np.ones(int(seq_length/8))
        begin = random.randint(1, 500)
        mask[-begin:] = 0
        return mask
    elif type == "y":
        y = np.zeros(7)
        return y
    
def GetRNAtype(dataset):
    with open(dataset, "r") as f1:
        string = f1.read()
    
    pattern = r"RNA_category:([^,\n]+)"
    RNA_types = re.findall(pattern, string)
    # print("RNA_types:", list(sorted(set(RNA_types))))
    return list(sorted(set(RNA_types)))



