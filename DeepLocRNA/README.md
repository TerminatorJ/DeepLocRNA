#requirement
#figure



if you want to train the model youself, please follow the following steps

## Step 1: generating the input data from your .fasta file 
## Input format
DeepLocRNA works on FASTA files, e.g.
```
>test1
ACTGCCGTATCGTAGCTAGCTAGTGATCGTAGCTACGTAGCTAGCTAGCTACGATCGTAGTCAGTCGTAGTACGTCA
>test2
ACACACATGAGCGATGTAGTCGATGATGCATCGACGATCGATCGAGCTACGTAGCATCGATCGATGCATCGACGTAG
```
save normal data
<pre>
python ./fine_tuning_deeprbploc_allRNA.py --dataset ./data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2_filtermilnc.fasta  
</pre>

save data with RNA tag

<pre>
python ./fine_tuning_deeprbploc_allRNA.py --dataset ./data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2_filtermilnc.fasta --RNA_tag
</pre>
Afterwards, there will be both "*_X_tag.npy" and "*_X.npy" in the "data/dataname/" folder.

Step 2: train the model

you have two options

first, you can use normal training strategy, using GPU to train the model

```
python ./fine_tuning_deeprbploc_allRNA.py --dataset ./data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2_filtermilnc.fasta --load_data --gpu_num 1 --species human --batch_size 8 --flatten_tag  --gradient_clip --loss_type BCE  --jobnum 001
```
Alternatively, DDP (data distributed parallel) strategy can be use to use multiple GPUs to train the model locally

```
python ./fine_tuning_deeprbploc_allRNA.py --dataset ./data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_deduplicated2_filtermilnc.fasta --load_data --gpu_num 1 --species human --batch_size 8 --flatten_tag  --gradient_clip --loss_type BCE  --jobnum 001 --DDP
```


Prediction mode.

There should prepare your input file to ".fasta" format

```
python fine_tuning_deeprbploc_allRNA_prediction.py --fasta ./example.fasta 
```


## IG scores calculation
# attribution config file
```
starts,ends
10,100
50,100
```

```
python fine_tuning_deeprbploc_allRNA_prediction.py --fasta ./example.fasta --att_config ./att_config.csv --plot True
```

arguments
Long                    |  Description
------------------------|  ------------
`--fasta`               |  Path to the input fasta file, formatted according to [the input format](#input-format).
`--att_config`          |  Path to the customized position of a specific sequence as `.csv`. formatted according to [att config format](#attribution-config-file)
`--plot`                |  Plot the attribution figures, this is mandatory if you want to visualize the explaination plots. Default: True



Alternatively, if you wish to get the precise nucleotide contribution, please choose "att_config" as True, and define the configure file yourself as "att_config.csv" before input in the input frame.

att_config can be defined as below (can also be downloaded from:https://github.com/TerminatorJ/DeepLocRNA/tree/main/DeepLocRNA/att_config.csv)
```
starts,ends
10,100
50,100
```
Where 10 to 100 is the interval that you want to get the attribution scores (Warning: you should not define the interval larger than 1000nt).

