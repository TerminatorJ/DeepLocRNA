#requirement
#figure



if you want to train the model youself, please follow the following steps

Step 1: generating the input data from your .fasta file 

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
python fine_tuning_deeprbploc_allRNA_prediction.py --fasta ./example.fasta --device cpu
```


