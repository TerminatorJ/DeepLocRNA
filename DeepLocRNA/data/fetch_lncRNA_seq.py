# %%
import pandas as pd
import numpy as np
import numpy as np
import torch

all_data = pd.read_table("./All_RNA_subcellular_localization_data.txt", sep = "\t")

###########Statistic of the human and mouse RAN data##############
ranked_locs = all_data[all_data["Species"] == "Homo sapiens"]["SubCellular_Localization"].value_counts(ascending=False).index
# print("The human dataset are shown as below:")
for loc in ranked_locs:
    human_data = all_data[all_data["Species"] == "Homo sapiens"]
    RNAs = dict(human_data[human_data["SubCellular_Localization"] == loc]["RNA_category"].value_counts(ascending=False))
    number = human_data[human_data["SubCellular_Localization"] == loc].shape[0]
    
    print("%s: %s, they come from %s" % (loc, number, RNAs))


#data glance
ranked_locs = all_data[all_data["Species"] == "Mus musculus"]["SubCellular_Localization"].value_counts(ascending=False).index
print("The mouse dataset has been shown as below")
for loc in ranked_locs:
    mouse_data = all_data[all_data["Species"] == "Mus musculus"]
    RNAs = dict(mouse_data[mouse_data["SubCellular_Localization"] == loc]["RNA_category"].value_counts(ascending=False))
    number = mouse_data[mouse_data["SubCellular_Localization"] == loc].shape[0]
    print("%s: %s, they come from %s" % (loc, number, RNAs))


def get_tag(label):
    if label in ["Nucleus", "Chromatin" , "Nucleoplasm", "Nucleolus", "Nuclear"]:
        label = "Nucleus"
    labels = np.array(["Nucleus","Exosome","Cytosol","Cytoplasm","Ribosome","Membrane","Endoplasmic reticulum"])
    locs = np.zeros(7)
    y = np.array(labels == label, dtype = "int")  
    # y_str = ''.join([str(i) for i in y])
    return list(y)

human_lnc = human_data[human_data["SubCellular_Localization"].isin(["Nucleus", "Chromatin" , "Nucleoplasm", "Nucleolus", "Nuclear","Exosome","Cytosol","Cytoplasm","Ribosome"]) & human_data["RNA_category"].isin(["lncRNA","lincRNA"])]
print("Counting the localization in the lncRNA:", human_lnc["SubCellular_Localization"].value_counts())

geneid_loc = {}
name_loc = {}
geneid_name = {}
all_c = 0
nan_c = 0
id_c = 0
id_symbol = {}
symbol_id = {}
# name_id = {}
# name_symbol = {}
for i in range(human_lnc.shape[0]):
    gene_name = list(human_lnc["Gene_Name"])[i]
    id_full = list(human_lnc["Gene_ID"])[i]
    loc = list(human_lnc["SubCellular_Localization"])[i]
    try:
        gene_id = id_full.split(":")[1]
        id_c+=1
        gene_symbol = list(human_lnc["Gene_symbol"])[i]
        id_symbol[gene_id] = gene_symbol
        symbol_id[gene_symbol] = gene_id
        geneid_loc.setdefault(gene_id,[]).append(get_tag(loc))
        geneid_name.setdefault(gene_id,[]).append(gene_name)# add to info later
    except:
        all_c+=1
        if str(id_full) == "nan":
            nan_c+=1
        name_loc[gene_name] =  loc
     
print("Number of id to loc:", id_c)
print("Number of non redundant id to loc:", len(list(geneid_loc.keys())))
print("Number of nan to loc:", nan_c)
print("Number of weired to loc:", all_c)
print("Number of non redundant weired to loc:", len(list(name_loc.keys())))

    

# Getting the main gene name based on alias names
# - From gene id to main name in order to get lncRNA sequence
# - Weired lncRNAs need to be filtered

# %%
gene_info = pd.read_table("Homo_sapiens.gene_info", index_col=0)
symbol2id = {}
id2name = {}
name2id = {}
# gene_info["GeneID"][:10]
name_failed = 0 

for geneid in geneid_loc.keys():
    try:
        name = gene_info[gene_info["GeneID"].isin([int(geneid)])]["Symbol"]
        id2name[geneid] = name.values[0]
        # name2id[name] = geneid
        name2id[name.values[0]] = geneid
    except:
        # id2name[geneid] = id_symbol[geneid]
        # name2id[id_symbol[geneid]] = geneid
        # print(geneid, "cannot get the gene name!!!")
        name_failed += 1

print(name_failed, "genes cannot get the gene name!!!")
print(len(list(geneid_loc.keys())), "genes are successful to fetch the gene names")


tags = ""
id_tags = {}
for geneid in list(geneid_loc.keys()):
    loc_ary = np.array(geneid_loc[geneid])
    multi_loc = loc_ary.sum(axis=0)
    multi_loc = np.array(multi_loc, dtype = "bool")
    multi_loc = np.array(multi_loc, dtype = "int")
    multi_loc = np.array(multi_loc, dtype = "str")
    multi_loc_tag = "".join(list(multi_loc))
    gene_names = "|".join(list(set(geneid_name[geneid])))
    tags += ">" + multi_loc_tag + "," + "Gene_ID:" + str(geneid) + "," + "Gene_Name:" + gene_names + "\n"
    tag = ">" + multi_loc_tag + "," + "Gene_ID:" + str(geneid) + "," + "Gene_Name:" + gene_names + "\n"
    id_tags[geneid] = tag.strip()
with open("./lncRNA_all_data_id.txt", "w") as f1:
    f1.write(tags)
    

name_seq = {}
name_info = {}


all_seq = pd.read_table("./All RNA sequence/human_RNA_sequence.txt")
# lnc_str = ""
got_id = {}
# non_got_id = []
name_seq = {}
name_info = {}
for i in range(all_seq.shape[0]):
    gene_id = all_seq["Gene_ID"][i].split(":")[1]
    seq = all_seq["Sequence"][i]
    if gene_id in list(id2name.keys()):
        # got_id[gene_id] = seq
        gene_name = id2name[gene_id]
        name_seq.setdefault(gene_name,[]).append(seq)
        name_info.setdefault(gene_name,[]).append(id_tags[gene_id]+"\n")

RNALocate_len = len(list(name_seq.keys()))
print(RNALocate_len, "genes have been extracted from RNALocatev2 sequence file")


map_c = 0
import mmap



with open("GRCh38_latest_rna.fna", "r") as f1:
    with mmap.mmap(f1.fileno(), 0, access=mmap.ACCESS_READ) as m:
        gene_name = ''
        for line in list(iter(m.readline, b'')):
            line = line.decode("utf-8")
            if line.startswith(">") == True:

                if gene_name != '':
                    if gene_name in list(id2name.values()):
                        name_seq.setdefault(gene_name,[]).append(seq)
                        geneid = name2id[gene_name]
                        name_info.setdefault(gene_name,[]).append(id_tags[geneid]+",%s\n" % refinfo)
                        map_c += 1
                refinfo = line.split("\n")[0][1:]
                # gene_name = line.split("(")[1].split(")")[0]
                gene_name = line.split(",")[0].split("(")[-1].split(")")[0]
                # print(gene_name)
                seq = ""
            else:
                seq += line.strip()

Ref_len = len(list(name_seq.keys()))
print(Ref_len- RNALocate_len, "genes have been extracted from Refseq sequence file")

with open("Homo_sapiens.GRCh38.ncrna.fa", "r") as f2:
    with mmap.mmap(f2.fileno(), 0, access=mmap.ACCESS_READ) as m:
        gene_name = ''
        for line in list(iter(m.readline, b'')):
            line = line.decode("utf-8")
            if line.startswith(">") == True:

                if gene_name != '':
                    if gene_name in list(id2name.values()):
                        print(gene_name)
                        name_seq.setdefault(gene_name,[]).append(seq)
                        geneid = name2id[gene_name]
                        name_info.setdefault(gene_name,[]).append(id_tags[geneid]+",%s\n" % refinfo)
                        map_c += 1
                
                # gene_name = line.split("(")[1].split(")")[0]
                if "gene_symbol" in line:
                    refinfo = line.split("\n")[0][1:]
                try:
                    
                    gene_name = line.split("gene_symbol:")[1].split(" ")[0]
                except:
                    # print(line)
                    continue
                # print(gene_name)
                seq = ""
            elif line.startswith("[") == True:
                continue
            else:
                seq += line.strip()


Ens_len = len(list(name_seq.keys()))
print(Ens_len- Ref_len, "genes have been extracted from Ensemble sequence file")
                
print(map_c, "genes are mapped to fna file")
print(len(list(name_seq.keys())) ,"genes are mapped to fna file")
print("Number of valid gene from RNALocate.v2:", len(list(geneid_loc.keys())))
print("Number of valid gene that can get the unique gene name based on file search:", len(list(id_symbol.keys())))
print("Number of valid gene that can get the sequences:", len(list(name_seq.keys())))

non_found_name = [name for name in name2id.keys() if name not in list(name_seq.keys())]
non_found_id = [name2id[name] for name in name2id.keys() if name not in list(name_seq.keys())]
print("There are some gene that cannot be found on the file, for example:", non_found_name, len(non_found_name))
print("There are some gene that cannot be found on the file, for example:", non_found_id, len(non_found_id))
   

print("Now i am going to fetch the losting sequence from the database")
with open("rnacentral_species_specific_ids.fasta", "r") as f3:
    with mmap.mmap(f3.fileno(), 0, access=mmap.ACCESS_READ) as m:
        gene_name = ''
        for line in list(iter(m.readline, b'')):
            line = line.decode("utf-8")
            if line.startswith(">") == True:

                if gene_name != '':
                    if gene_name in list(id2name.values()):
                        name_seq.setdefault(gene_name,[]).append(seq)
                        geneid = name2id[gene_name]
                        name_info.setdefault(gene_name,[]).append(id_tags[geneid]+",%s\n" % refinfo)
                        map_c += 1
                # refinfo = line.split("\n")[0][1:]
                # gene_name = line.split("(")[1].split(")")[0]
                line
                try:
                    refinfo = line.split("\n")[0][1:]
                    gene_name = line.split(" ")[-1].strip()
                except:
                    # print(line)
                    continue
                # print(gene_name)
                seq = ""
            elif line.startswith("[") == True:
                continue
            else:
                seq += line.strip()

print("Number of valid gene from RNALocate.v2:", len(list(geneid_loc.keys())))
print("Number of valid gene that can get the unique gene name based on file search:", len(list(id2name.keys())))
print("Number of valid gene that can get the sequences:", len(list(name_seq.keys())))



# %%
# print(name_info)

# print("name_info HOXC-AS1", name_info["HOXC-AS1"])
# print("name_seq:",name_seq)
# print("name_info", name_info)

new_str = ""
deeplncloc_data = pd.read_table("/home/sxr280/DeepRBPLoc/new_data/data.txt", names = ["gene","seq","loc"])
for i in range(deeplncloc_data.shape[0]):
    gene = deeplncloc_data["gene"][i]
    seq = deeplncloc_data["seq"][i]
    loc = deeplncloc_data["loc"][i]
    tag = ">" + "".join([str(i) for i in get_tag(loc)]) + "," + gene + "\n" + seq + "\n"
    new_str += tag


items = ""
for gene in name_seq.keys():
    seq = name_seq[gene]
    longest_seq = max(seq, key=len)
    # print(longest_seq)
    idx = seq.index(longest_seq)
    info = name_info[gene]
    # print(idx)
    longest_info = info[idx]
    items += "%s%s\n" % (longest_info,longest_seq)
    # print(longest_info,longest_seq)
# print(items)


# with open("./lncRNA_all_data_seq.fasta", "w") as f1:
#     f1.write(items)


with open("./lncRNA_all_data_seq_includingmouse.fasta", "w") as f1:
    f1.write(items+new_str)
    

# %%



