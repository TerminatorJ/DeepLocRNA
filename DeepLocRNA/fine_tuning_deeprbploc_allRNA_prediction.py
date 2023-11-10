# %%
from captum.attr import IntegratedGradients
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from preprocessing import *
sys.path.insert(0, "./data")
import torch
from multihead_attention_model_torch import *
import gin
from plot import *

import argparse
import pandas as pd
from model_explaination import *
import warnings

# Filter out the UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
refs = np.array(["Nucleus","Exosome","Cytosol","Cytoplasm","Ribosome","Membrane","Endoplasmic reticulum", "Microvesicle", "Mitochondrion"])
#thredshold of each compartment
#0.7551 for the nucleus, 0.9796 for exosome, 0.2245 for cytosol, 0.2857 for ribosome, 0.3061 for membrane, and 0.1837 for the ER.
all_thredsholds = {"Human":{"mRNA":{"Nucleus": 0.7551, "Exosome": 0.9796, "Cytosol": 0.2245, "Ribosome": 0.2857, "Membrane": 0.3061, "Endoplasmic reticulum": 0.1837},
                "miRNA": {"Nucleus": 0.0204, "Exosome": 0.9592, "Cytoplasm": 0.0204, "Microvesicle": 0.8776, "Mitochondrion": 0.0204},
                "lncRNA": {"Nucleus": 0.1020, "Exosome": 0.9796, "Cytosol": 0.2041, "Membrane": 0.0816},
                "snoRNA": {"Nucleus": 0.5714, "Exosome": 0.0000, "Cytoplasm": 0.0612, "Microvesicle": 0.0000}}, 
                "Mouse": {"mRNA":{"Nucleus": 0.4694, "Exosome": 0.6327, "Cytoplasm": 0.2653}, 
                "miRNA":{"Nucleus": 0.1837, "Exosome": 0.7347, "Mitochondrion": 0.4082},
                "lncRNA": {"Nucleus": 0.3673, "Exosome": 0.1837, "Cytoplasm": 0.2245}}}


def pass_thred(pred, thred):
    #get meassured localizations
    pred_str = []
    # print(pred, thred)
    locs = list(thred.keys())
    # print("locs:", locs)
    loc_idx = np.where(np.isin(refs, locs))[0]
    # print("loc_idx", loc_idx)
    loc_pred = pred[loc_idx]
    # print("loc_pred", loc_pred)
    for idx, v in enumerate(loc_pred):

        if v > thred[locs[idx]]:
            # print(v, thred[locs[idx]])
            pred_str.append(locs[idx])
    pred_str = "/".join(pred_str)
    #replace cytoplasm as cytosol 
    pred_str = pred_str.replace("Cytoplasm", "Cytosol")
    return pred_str

def get_att(att):
    att = att.squeeze()
    att = np.transpose(att, (1, 0))
    flat_att = att.squeeze().sum(axis=1)
    return att, flat_att


def squeeze(array):
    if isinstance(array, torch.Tensor):
        return array.unsqueeze(0)
    else:
        return np.expand_dims(array, 0)


def predict(fasta, rna_types, batch_size = 2, plot = "False", att_config = None, species = "Human"):
    #get specific threshold
    type_thred = all_thredsholds[species][rna_types]
    #generating the data
    input_types = rna_types
    X, mask_label, ids = preprocess_data2(left=4000, right=4000, dataset=fasta, padmod="after",pooling_size=8, foldnum=1, pooling=True, RNA_type = "singleRNA", RNA_tag = False, input_types = input_types, species = species)
    X_tag = preprocess_data2(left=4000, right=4000, dataset=fasta, padmod="after",pooling_size=8, foldnum=1, pooling=True, RNA_type = "singleRNA", RNA_tag = True, input_types = input_types, species = species)[0]

    #building dataloader
    X = torch.from_numpy(X).to(device, torch.float)
    X_tag = torch.from_numpy(X_tag).to(device, torch.float)
    mask_label = torch.from_numpy(mask_label).to(device, torch.float)


    loaded_dataset = torch.utils.data.TensorDataset(X, X_tag, mask_label)
    dataloader = torch.utils.data.DataLoader(loaded_dataset, batch_size = batch_size, shuffle = False)



    #loading the model
    current_path = os.getcwd()
    if species == "Human":
        ckp_path = os.path.join(current_path, "Result", "allRNA_finetuning", "checkpoints_allRNA_False_20_0_human_True_17709_best.ckpt")
    elif species == "Mouse":
        ckp_path = os.path.join(current_path, "Result", "allRNA_finetuning", "checkpoints_allRNA_False_20_0_mouse_True_13107_best.ckpt")


    hyperparams_1 = {
                'fc_dim': 500,
                'weight_decay': 1e-5,
                'attention': True,####need to be changed
                'lr': 0.005,
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
                "RNA_type" : "allRNA",
                "dim_attention" : 50,
                "species": species
                }
       
    gin.parse_config_file('./config.gin')
    DeepLocRNA_model = myModel1(**hyperparams_1)
    DeepLocRNA_model.network.att = False
    checkpoint = torch.load(ckp_path, map_location=torch.device(device))
    model_state = checkpoint['state_dict']
    DeepLocRNA_model.load_state_dict(model_state)
    DeepLocRNA_model.eval()
    DeepLocRNA_model.to(device = device)

    all_y_pred_list = []
    counter = 0
    items = 0
    for i, batch in enumerate(dataloader):

        counter = (i+1)*batch_size
        assert counter <= 200, "WARNING: The number of input sequences should not larger than 200!!!"
        X, X_tag, mask = batch
        y_pred = DeepLocRNA_model.forward(X, mask, X_tag)
        y_pred = y_pred.detach().cpu().numpy()
        all_y_pred_list.append(y_pred)

        if plot == "True":
            ig = IntegratedGradients(DeepLocRNA_model)
            #get predicted label
            batch_str = [pass_thred(pred, type_thred) for pred in y_pred]
            X = X.long()
            #switch to att mode
            DeepLocRNA_model.network.att = True
            embedding_output = DeepLocRNA_model.network.embedding_layer(X)#[8000, 4]
            embedding_output = embedding_output.transpose(1,2)#[4, 8000]         
            embedding_output = embedding_output.to(torch.float32)
            for idx, pred in enumerate(y_pred):
                items += 1
                embedding = embedding_output[idx].unsqueeze(0)
                m = mask[idx].unsqueeze(0)
                x_tag = X_tag[idx].unsqueeze(0)
                length = int(torch.sum(mask[idx]).item())
                id = ids[idx]
                for t in batch_str[idx].split("/"):
                    t_num = list(refs).index(t)
                    attributions = ig.attribute(inputs=(embedding), target=t_num, n_steps=25, additional_forward_args = (m,x_tag), return_convergence_delta = False, internal_batch_size = 8)
                    attributions = attributions.detach().cpu().numpy()
                    att, flat_att = get_att(attributions)
                    fig1 = plot_line(flat_att[:length])
                    fig1.savefig("./motif_line_plot_%s_%s.png" % (items, t), dpi=60)
                    if att_config is not None:
                        att_cfg = pd.read_csv(att_config, skipinitialspace=True)
                        s = att_cfg["starts"][idx]
                        e = att_cfg["ends"][idx]
                        assert e-s <= 1000, "the defined motif should shorter than 1000nt"
                        fig2 = plot_motif(att[s:e+1], figsize=(16, 6), start = s)#to show complete x-axis
                        fig2.savefig("./motif_log_%s_%s.png" % (items, t), dpi=60)
                    


    # Convert the list of arrays to a single NumPy array
    all_y_pred = np.vstack(all_y_pred_list)
    #getting the values that pass the thredholds
    # print(all_y_pred)
    results_str = [pass_thred(pred, type_thred) for pred in all_y_pred]
    # print(results_str)

    result_df = pd.DataFrame(data = all_y_pred, columns = refs, index = ids)
    #embedding the prediction string
    result_df["Prediction"] = results_str
    result_df.to_csv(os.path.join(current_path, "output.csv"))




if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', type=str, default=None, help='The input fasta to be predicted')
    parser.add_argument('--rna_types', type=str, default=None, help='The RNA types to be predicted')
    parser.add_argument('--plot', type=str, default="False", help='Whether generating the attribution plot')
    parser.add_argument('--att_config', type=str, default=None, help='The file that is used to define in which position to display IG score')
    parser.add_argument('--species', type=str, default="Human", help='The species you want to predict')
    args = parser.parse_args()

    predict(fasta = args.fasta, rna_types = args.rna_types, plot = args.plot, att_config = args.att_config, species = args.species)
    print("Please download the output by pressing the download button, you will find the output.csv file afterwards!!!")


#python fine_tuning_deeprbploc_allRNA_prediction.py --fasta ./example.fasta --device cpu








