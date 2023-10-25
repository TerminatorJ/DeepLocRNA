# %%

import sys
from preprocessing import *
sys.path.insert(0, "./data")
import torch
from multihead_attention_model_torch import *
import gin

import argparse
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(fasta, batch_size = 2):

    #generating the data
    X, mask_label, ids = preprocess_data2(left=4000, right=4000, dataset=fasta, padmod="after",pooling_size=8, foldnum=1, pooling=True, RNA_type = "RNA", RNA_tag = False)
    X_tag = preprocess_data2(left=4000, right=4000, dataset=fasta, padmod="after",pooling_size=8, foldnum=1, pooling=True, RNA_type = "RNA", RNA_tag = False)[0]

    #building dataloader
    X = torch.from_numpy(X).to(device, torch.float)
    X_tag = torch.from_numpy(X_tag).to(device, torch.float)
    mask_label = torch.from_numpy(mask_label).to(device, torch.float)


    train_dataset = torch.utils.data.TensorDataset(X, X_tag, mask_label)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = False)



    #loading the model
    current_path = os.getcwd()
    ckp_path = os.path.join(current_path, "Result", "allRNA_finetuning", "checkpoints_allRNA_False_20_0_human_True_21482_best.ckpt")
    # DeepLocRNA_model = torch.load(model_path, map_location=torch.device(device))
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
                "dim_attention" : 50
                }
       
    gin.parse_config_file('./config.gin')
    DeepLocRNA_model = myModel1(**hyperparams_1)
    checkpoint = torch.load(ckp_path, map_location=torch.device(device))
    model_state = checkpoint['state_dict']
    DeepLocRNA_model.load_state_dict(model_state)

    
    #doing the prediction
    #doing the prediction
    all_y_pred_list = []
    counter = 0
    for i, batch in enumerate(dataloader):

        counter = (i+1)*batch_size
        assert counter <= 200, "WARNING: The number of input sequences should not larger than 200!!!"
        X, X_tag, mask = batch
        y_pred = DeepLocRNA_model.forward(X, mask, X_tag)
        y_pred = y_pred.detach().cpu().numpy()
        all_y_pred_list.append(y_pred)

    # Convert the list of arrays to a single NumPy array
    all_y_pred = np.vstack(all_y_pred_list)
    
    refs = np.array(["Nucleus","Exosome","Cytosol","Cytoplasm","Ribosome","Membrane","Endoplasmic reticulum", "Microvesicle", "Mitochondrion"])
    # print(ids,all_y_pred,refs)
    result_df = pd.DataFrame(data = all_y_pred, columns = refs, index = ids)
    
    result_df.to_csv(os.path.join(current_path, "output.csv"))

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', type=str, default=None, help='The input fasta to be predicted')
    # parser.add_argument('--device', type=str, default="cpu", help='The device to process the sequence prediction')
    args = parser.parse_args()

    predict(args.fasta)
    print("Please download the output by pressing the download button, you will find the output.csv file afterwards!!!")


#python fine_tuning_deeprbploc_allRNA_prediction.py --fasta ./example.fasta --device cpu








