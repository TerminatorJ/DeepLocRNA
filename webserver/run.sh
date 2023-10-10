#!/bin/bash

mkdir output


python ./DeepLocRNA/fine_tuning_deeprbploc_allRNA_prediction.py --fasta ./example.fasta --device cpu
#python generate_output.py

cat ./DeepLocRNA/DeepLocRNA/output.csv