# %%
import pandas as pd
import numpy as np
import numpy as np
import torch


# Testing the ratio of the test data

# %%
#loading the benchmark dataset -- training data
# y = np.load("/home/sxr280/DeepRBPLoc/testdata/Test_fold0_y.npy")
# test_num = y.shape[0]
# np.sum(y, axis=0)/test_num

# %% [markdown]
# Loading all the data in one line

# %%
all_data = pd.read_table("./All_RNA_subcellular_localization_data.txt", sep = "\t")



# %% [markdown]
# To see how many data we have in total for Human

# %%
#data glance
ranked_locs = all_data[all_data["Species"] == "Homo sapiens"]["SubCellular_Localization"].value_counts(ascending=False).index
# print("The human dataset are shown as below:")
for loc in ranked_locs:
    human_data = all_data[all_data["Species"] == "Homo sapiens"]
    RNAs = dict(human_data[human_data["SubCellular_Localization"] == loc]["RNA_category"].value_counts(ascending=False))
    number = human_data[human_data["SubCellular_Localization"] == loc].shape[0]
    
    print("%s: %s, they come from %s" % (loc, number, RNAs))


# %% [markdown]
# how many data we have for Mouse

# %%
#data glance
ranked_locs = all_data[all_data["Species"] == "Mus musculus"]["SubCellular_Localization"].value_counts(ascending=False).index
print("The mouse dataset has been shown as below")
for loc in ranked_locs:
    mouse_data = all_data[all_data["Species"] == "Mus musculus"]
    RNAs = dict(mouse_data[mouse_data["SubCellular_Localization"] == loc]["RNA_category"].value_counts(ascending=False))
    number = mouse_data[mouse_data["SubCellular_Localization"] == loc].shape[0]
    print("%s: %s, they come from %s" % (loc, number, RNAs))


# %%
#filter the human dataset
#we want all the samples that distinguish the nuclear/nucleus
# human_data = all_data[(all_data["Species"] == "Homo sapiens") & (all_data["RNA_category"] == "mRNA") & (all_data["SubCellular_Localization"].isin(["Nucleus", "Chromatin" , "Nucleoplasm", "Nucleolus", "Nuclear", "Exosome","Cytosol","Cytoplasm","Ribosome","Membrane","Endoplasmic reticulum"]))]
# human_data


# %% [markdown]
# Extract the lncRNA data set from human set
# - the finalized fasta file must be like >1100000,ACCNUM:NM_001672,Gene_ID:434,Gene_Name:ASIP
# - filter the independent dataset

# %%
def get_tag(label):
    if label in ["Nucleus", "Chromatin" , "Nucleoplasm", "Nucleolus", "Nuclear"]:
        label = "Nucleus"
    labels = np.array(["Nucleus","Exosome","Cytosol","Cytoplasm","Ribosome","Membrane","Endoplasmic reticulum","Microvesicle", "Mitochondrion"])
    locs = np.zeros(7)
    y = np.array(labels == label, dtype = "int")  
    # y_str = ''.join([str(i) for i in y])
    return list(y)


# %%


# %%
human_mi = human_data[human_data["SubCellular_Localization"].isin(["Nucleus", "Chromatin" , "Nucleoplasm", "Nucleolus", "Nuclear","Exosome","Cytosol","Cytoplasm","Ribosome","Microvesicle", "Mitochondrion"]) & human_data["RNA_category"].isin(["miRNA"])]

print("Counting the localization in the miRNA:", human_mi["SubCellular_Localization"].value_counts())

id_loc = {}
geneid_name = {}
# idf = []
# name_id = {}
# name_symbol = {}
for i in range(human_mi.shape[0]):
    gene_name = list(human_mi["Gene_Name"])[i]
    id_full = str(list(human_mi["Gene_ID"])[i])
    # print(gene_name)
    # idf.append(gene_name.split("|") + id_full.split("|"))
    loc = list(human_mi["SubCellular_Localization"])[i]
    
    id_loc.setdefault(id_full,[]).append(get_tag(loc))
    if loc in ["Mitochondrion"]:
        print(id_full, gene_name,loc,get_tag(loc))
    geneid_name.setdefault(id_full,[]).append(gene_name)

print("Number of non redundant gene name to loc:", len(list(id_loc.keys())))



# # %% [markdown]
# # Adding the multitag to each lncRNA

# # %%
tags = ""
id_tags = {}
for geneid in list(id_loc.keys()):
    loc_ary = np.array(id_loc[geneid])
    multi_loc = loc_ary.sum(axis=0)
    multi_loc = np.array(multi_loc, dtype = "bool")
    multi_loc = np.array(multi_loc, dtype = "int")
    multi_loc = np.array(multi_loc, dtype = "str")
    multi_loc_tag = "".join(list(multi_loc))
    name = "|".join(list(set(geneid_name[geneid])))
    # print(name)
    # print(len(str(geneid)))
    # if len(str(geneid)) > 28:
    # print(str(geneid))
    tags += ">" + multi_loc_tag + "," + "Gene_ID:" + str(geneid) + "," + "Gene_Name:" + name + "\n"
    tag = ">" + multi_loc_tag + "," + "Gene_ID:" + str(geneid) + "," + "Gene_Name:" + name + "\n"
    id_tags[geneid] = tag.strip()
with open("./miRNA_all_data_id.txt", "w") as f1:
    f1.write(tags)

#do the simple match to the two files

id_seq = {}
id_info = {}
miRNA_df = pd.read_table("/home/sxr280/DeepRBPLoc/new_data/All RNA sequence/all_miRNA_sequence.txt")
for i in range(miRNA_df.shape[0]):
    symbol = miRNA_df["Gene_Symbol"][i]
    gene_id = miRNA_df["Gene_ID"][i]
    seq = miRNA_df["Sequence"][i]
    if gene_id in id_loc.keys():
        id_seq.setdefault(gene_id,[]).append(seq)
        id_info.setdefault(gene_id,[]).append(id_tags[gene_id]+"\n")
leak = [i for i in id_loc.keys() if i not in id_seq.keys()]

print(len(leak),"genes have been leaked", len(list(id_seq.keys())), "genes have been fetched")
items = ""
for geneid in id_seq.keys():
    seq = id_seq[geneid]
    longest_seq = max(seq, key=len)
    idx = seq.index(longest_seq)
    info = id_info[geneid]
    longest_info = info[idx]
    items += longest_info+longest_seq+"\n"
    

with open("./miRNA_all_data_seq.fasta", "w") as f1:
    f1.write(items)




