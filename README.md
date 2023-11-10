![DeepLocRNA](webserver/assets/Figure1small.png)
# DeepLocRNA

## Introduction
DeeplocRNA is a deep neural network that predict the RNA localization, enabling the prediction across 4 RNA species (mRNA, miRNA, lncRNA, snoRNA) and different species (Human and Mouse). 



## Train the model
if you want to train the model youself, please follow the following steps

### Step 1: Preparing your FASTA file
#### Input format
DeepLocRNA works on FASTA files, e.g.
```
>test1
ACTGCCGTATCGTAGCTAGCTAGTGATCGTAGCTACGTAGCTAGCTAGCTACGATCGTAGTCAGTCGTAGTACGTCA
>test2
ACACACATGAGCGATGTAGTCGATGATGCATCGACGATCGATCGAGCTACGTAGCATCGATCGATGCATCGACGTAG
```
### Step 2: Save encoded data

<pre>
python ./DeepLocRNA/fine_tuning_deeprbploc_allRNA.py --dataset ./data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2_filtermilnc.fasta  
</pre>

### Step 2: Save encoded tagged data
In order to do multiple RNA prediction, we will generate tag for each RNA species
<pre>
python ./DeepLocRNA/fine_tuning_deeprbploc_allRNA.py --dataset ./data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2_filtermilnc.fasta --RNA_tag
</pre>
Afterwards, there will be both "*_X_tag.npy" and "*_X.npy" in the "./DeepLocRNA/data/dataname/" folder.

### Step 3: Training the model

you have two options

first, you can use standard training strategy, using GPUs to train the model

```
python ./fine_tuning_deeprbploc_allRNA.py --dataset ./data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2_filtermilnc.fasta --load_data --gpu_num 1 --species human --batch_size 8 --flatten_tag  --gradient_clip --loss_type BCE  --jobnum 001
```
Alternatively, DDP (data distributed parallel) strategy can be use to use multiple GPUs to train the model locally

```
python ./fine_tuning_deeprbploc_allRNA.py --dataset ./data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2_filtermilnc.fasta --load_data --gpu_num 1 --species human --batch_size 8 --flatten_tag  --gradient_clip --loss_type BCE  --jobnum 001 --DDP
```


## Model Prediction.

There should prepare your input file to ".fasta"(#Input-format) format

```
python ./DeepLocRNA/fine_tuning_deeprbploc_allRNA_prediction.py --fasta ./example.fasta 
```
Alternatively, you can also use our online webserver if you only have a couple sequences to be predicted ()

## IG scores calculation

### attribution config file
```
starts,ends
10,100
50,100
```
Where 10 to 100 is the interval that you want to get the attribution scores.
```
python fine_tuning_deeprbploc_allRNA_prediction.py --fasta ./example.fasta --att_config ./att_config.csv --plot True
```
If you wish to get the precise nucleotide contribution, please choose "--plot" as True, and define the configure file yourself as "att_config.csv" before input in the input frame.


arguments
Long                    |  Description
------------------------|  ------------
`--fasta`               |  Path to the input fasta file, formatted according to [the input format](#input-format).
`--att_config`          |  Path to the customized position of a specific sequence as `.csv`. formatted according to [att config format](#attribution-config-file)
`--plot`                |  Plot the attribution figures, this is mandatory if you want to visualize the explaination plots. Default: True










