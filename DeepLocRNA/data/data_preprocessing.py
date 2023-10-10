# %%
import pandas as pd
import numpy as np
import numpy as np
import torch

# %% [markdown]
# Testing the ratio of the test data

# %%
#loading the benchmark dataset -- training data
y = np.load("/home/sxr280/DeepRBPLoc/testdata/Test_fold0_y.npy")
test_num = y.shape[0]
np.sum(y, axis=0)/test_num

# %% [markdown]
# Loading all the data in one line

# %%
all_data = pd.read_table("./All_RNA_subcellular_localization_data.txt", sep = "\t")



# %% [markdown]
# To see how many data we have in total for Human

# %%
#data glance
ranked_locs = all_data[all_data["Species"] == "Homo sapiens"]["SubCellular_Localization"].value_counts(ascending=False).index
print("The human dataset are shown as below:")
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
    labels = np.array(["Nucleus","Exsome","Cytosol","Cytoplasm","Ribosome","Membrane","Endoplasmic reticulum"])
    locs = np.zeros(7)
    y = np.array(labels == label, dtype = "int")  
    y_str = ''.join([str(i) for i in y])
    return y_str


# %%
human_lnc = human_data[human_data["SubCellular_Localization"].isin(["Nucleus", "Chromatin" , "Nucleoplasm", "Nucleolus", "Nuclear","Exosome","Cytosol","Cytoplasm","Ribosome"]) & human_data["RNA_category"].isin(["lncRNA"])]
print("Counting the localization in the lncRNA:", human_lnc["SubCellular_Localization"].value_counts())
gene_id_lst = []
unique_gene = []
tags = ''
for i in range(human_lnc.shape[0]):
    gene_id = list(human_lnc["Gene_ID"])[i]
    gene_name = list(human_lnc["Gene_Name"])[i]
    gene_symbol = list(human_lnc["Gene_symbol"])[i]
    loc = list(human_lnc["SubCellular_Localization"])[i]
    ini_loc = loc
    loc_tag = get_tag(loc)
    unique_id = str(gene_name)+":"+str(loc)
    if unique_id not in unique_gene:
        unique_gene.append(unique_id)
        if loc_tag != "0000000":
            tags += ">" + loc_tag + "," + ini_loc + "," + "Gene_ID:" + str(gene_id) + "," + "Gene_Name:" + str(gene_name) + "\n"
        #excluding the independent test set
        # gene_id.isin()
        # Gene_symbol.isin()
    #saving the lncRNA dataset
            with open("./lncRNA_all_data_id.txt", "w") as f1:
                f1.write(tags)
    else:
        continue

    



# %%
human_lnc[human_lnc["Gene_Name"] == "ANRIL"]

# %% [markdown]
# In order to match the sequence with the independent lncRNA dataset, we need to get access to NCBI to get the genomics intervals.
# - Trhy to get access via Bio.Entrez

# %%
from Bio import Entrez
from Bio import SeqIO
Entrez.email = 'wangjun19950708@gmail.com'
handle = Entrez.efetch(db="nucleotide", id="NC_000012", rettype="fasta", retmode="text")
# print(handle.readline().strip())
record = SeqIO.read(handle, "fasta")
# print(record.__dict__)
print(record.seq)

# %%
handle.__dict__

# %%
import xml.etree.cElementTree as ElementTree
handle = Entrez.esearch(db='nucleotide', term="HOTAIR", retmax=1)
print(handle)
root = ElementTree.fromstring(handle.read())
print(dict(root))
id_number = root.find("IdList/Id").text
print(id_number)

# %%
print(record.id)

# %%
print(record.description)

# %%
for feature in record.features:
    print(feature.location) 

# %%
record.features

# %%



