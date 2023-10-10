
for num in {0..8}
do
    echo "do: $num"
    python ./scripts/seed-motif-alignment.py /home/sxr280/DeepRBPLoc/motif_analysis/DeepRBPLoc_motif/kmer_5target_"$num"_df.txt --min-support 50 -o kmer_5target_"$num"
done