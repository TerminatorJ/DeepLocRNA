# DeepLocRNA

DeeplocRNA is a deep neural network that predict the RNA localization, enabling the prediction across 5 RNA species (mRNA, miRNA, lncRNA, snRNA, snoRNA) and different species (Human and Mouse).  

If you are also interested in the protein localization, please refer to our protein version:  
https://services.healthtech.dtu.dk/services/DeepLoc-2.0/

In this webserver, we support the input file in a ".fasta" format. Because of the limited computational resource, this webserver only supports up to 200 sequences.  

Explaination figures are also supported to be downloaded.   
If you choose "plot" as True, a full-length attribution will be generated for you.  

Alternatively, if you wish to get the precise nucleotide contribution, please choose "att_config" as True, and define the configure file yourself as "att_config.csv" before input in the input frame.

att_config can be defined as below (can also be downloaded from:https://github.com/TerminatorJ/DeepLocRNA/tree/main/DeepLocRNA/att_config.csv)
```
starts,ends
10,100
50,100
```
Where 10 to 100 is the interval that you want to get the attribution scores (Warning: you should not define the interval larger than 1000nt).


Alternatively, if you want to predict a large number of RNA sequence, please download our standalone tool by  
```
pip install DeepLocRNA
```

More instructions of the standalone tool can refer to our git repository as: 
https://github.com/TerminatorJ/DeepLocRNA  
# Concept image
![DeepLocRNA](assets/Figure1small.png)


