import torch
import captum
from captum.attr import IntegratedGradients
import os, sys
import os
import torch
import pickle
from collections import OrderedDict
import sys
sys.path.insert(0, "../../")
sys.path.insert(0, "../../new_data")
sys.path.insert(0, "../../attention_weight_analysis")
from fine_tuning_deeprbploc_allRNA import *
from multihead_attention_model_torch import *
from torch.utils.data import DataLoader, ConcatDataset
import time
import argparse
import os
import logomaker
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class motif_analysis:
    def __init__(self, fold = 0, layer = 20, batch_size = 4, dataset = None, test_mode = False, fasta = None, dataloader = None, model = None, ids = None):
        # device = device
        self.fold = fold
        self.layer = layer
        self.batch_size = batch_size
        self.dataset = dataset
        self.test_mode = test_mode
        self.fasta = fasta
        self.dataloader = dataloader
        self.model = model
        self.ids = ids
  

    def input_file(self, batch_size = 8):
        X, mask_label, ids = preprocess_data2(left=4000, right=4000, dataset=self.fasta, padmod="after",pooling_size=8, foldnum=1, pooling=True, RNA_type = "singleRNA", RNA_tag = False, input_types = "mRNA")
        X_tag = preprocess_data2(left=4000, right=4000, dataset=self.fasta, padmod="after",pooling_size=8, foldnum=1, pooling=True, RNA_type = "singleRNA", RNA_tag = True, input_types = "mRNA")[0]
        X = torch.from_numpy(X).to(device, torch.float)
        X_tag = torch.from_numpy(X_tag).to(device, torch.float)
        # y = torch.from_numpy(y).to(device, torch.float)
        mask_label = torch.from_numpy(mask_label).to(device, torch.float)
        train_dataset = torch.utils.data.TensorDataset(X, X_tag, mask_label)
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
        return dataloader,ids
    def IG_calculation(self, target, n_steps = 50):
        target_names = ["Nucleus","Exosome","Cytosol","Cytoplasm","Ribosome","Membrane","Endoplasmic_reticulum", "Microvesicle", "Mitochondrion"]
        target_name = target_names[target]
        ig = IntegratedGradients(self.model)
        all_att = torch.tensor([]).to(device)
        X_all = torch.tensor([]).to(device)

        combined_dataloader,ids = self.dataloader, self.ids
        all_id = ids
        if not self.test_mode:
            for i, batch in enumerate(combined_dataloader):
                X, X_tag, X_mask = batch
                X = X.to(device)
                X_tag = X_tag.to(device)
                X_mask = X_mask.to(device)
                #processing the input after embedding
                X = X.long()
                embedding_output = self.model.network.embedding_layer(X)#[8000, 4]
                embedding_output = embedding_output.transpose(1,2)#[4, 8000]         
                embedding_output = embedding_output.to(torch.float32)
                attributions = ig.attribute(inputs=(embedding_output), target=target, n_steps=n_steps, additional_forward_args = (X_mask,X_tag), return_convergence_delta = False, internal_batch_size = 8)
                all_att = torch.cat((all_att,attributions), 0)
                X_all = torch.cat((X_all,X), 0)

            all_att = all_att.cpu().detach().numpy()
            print(all_att.shape)
            all_att, all_flat_att = self.get_att(all_att)
            fig1 = self.plot_motif(all_att, figsize=(16, 6))
            fig1.savefig("./motif_log_%s.png" % target_name, dpi=300)
            fig2 = self.plot_line(all_flat_att)
            fig2.savefig("./motif_line_plot_%s.png" % target_name, dpi=300)

                
        else:
            X, X_tag, X_mask, y = next(iter(combined_dataloader))
            X = X.to(device)
            X_tag = X_tag.to(device)
            X_mask = X_mask.to(device)
            #processing the input after embedding
            X = X.long()
            # print("X device:", X.device)
            embedding_output = self.model.network.embedding_layer(X)#[8000, 4]
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

    def plot_motif(self, motif_array, sigma=['A', 'C', 'G', 'U'], ax=None, figsize=(16, 9/2), title=None, title_x=.50, title_y=1.05, ylab=None, start = None, fontsize = 16):
        if ax is None:
            fig, axs = plt.subplots(1, 1, figsize=figsize)
        df = pd.DataFrame(motif_array, columns=sigma)
        logomaker.Logo(df, ax=axs) # , shade_below=.5 fade_below=.5, 
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        
        if title is not None:
            axs.set_title(title, x = title_x, y = title_y, fontsize=16)
            
        if ylab is not None:
            axs.set_ylabel(ylab, fontsize=16)
        
        #axs.set_yticks(ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=26)
        # axs.set_xticks(ticks = [i for i in range(motif_array.shape[0])])
        # axs.set_xticklabels([str(i + 2854) for i in range(motif_array.shape[0])], rotation=45, ha='right')
        x_positions = range(0, motif_array.shape[0], 10)  # Define the x-tick positions every 100 numbers
        x_labels = [str(x+start) for x in x_positions]  # Create x-tick labels
        # print(x_labels)
        axs.set_xticks(ticks=x_positions)
        axs.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=fontsize)  
        axs.set_yticklabels(axs.get_yticklabels(), fontsize=fontsize)
        if ax is None:
            return fig
    def plot_line(self, flat_att):
        fig, axs = plt.subplots(1, 1, figsize=(10, 6))
        plt.plot([i for i in range(len(flat_att))], flat_att, linestyle='-', color = "blue")  # 'o' for markers, '-' for line style
        plt.title("Length vs IG score")
        plt.xlabel("Positions", fontsize=16)
        plt.ylabel("IG score", fontsize=16)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)

        axs.set_yticklabels(axs.get_yticklabels(), fontsize=16)
        axs.set_xticklabels(axs.get_xticklabels(), fontsize=16)
        return fig
    
    def get_att(self,att):
        att = att.squeeze()
        att = np.transpose(att, (1, 0))
        flat_att = att.squeeze().sum(axis=1)
        return att, flat_att


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
    # print("flat_att", flat_att.shape, flat_att[:,6000:6050])
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
        # print(item)
        # print(motifs[idx])
        # print(flat_att_val[idx])
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


def main(k_mer=5, target = 0, test_mode = False, fasta = None, get_max = False):
    target_names = ["Nucleus","Exosome","Cytosol","Cytoplasm","Ribosome","Membrane","Endoplasmic_reticulum", "Microvesicle", "Mitochondrion"]
    target_name = target_names[target]

    #getting the proper input dataset
    dataset = "./data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_pooled_deduplicated3_filtermilncsnsno.fasta"

    
    
    ana = motif_analysis(fold = 3, layer = 20, batch_size = 32, dataset = dataset, test_mode = test_mode, fasta = fasta)
    all_att,X_all,all_id = ana.IG_calculation(target = target, n_steps = 25)
    if get_max:
        prefix = "explaination_kmer_" + str(k_mer) + "target_" + str(target)
        motifs, flat_att_p, flat_att_val = get_max_motif(all_att, X_all, all_id, k_mer, prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=3, help='which compartment you want to calculate')
    parser.add_argument('--fasta', type=str, default=None, help="the fasta file you input for external genes")
    # parser.add_argument('--t', type=int, default=3, help='which compartment you want to calculate')
    
    args = parser.parse_args()
    start_time = time.time()
    print("running target:", args.target)
    main(k_mer=5, target = args.target, test_mode = False, fasta=args.fasta)
    # main(k_mer=5, target = 3, test_mode = False, fasta="/home/sxr280/DeepRBPLoc/attention_weight_analysis/ACTB.fasta")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")


# %%


