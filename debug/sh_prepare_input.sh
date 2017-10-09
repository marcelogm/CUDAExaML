#!/bin/bash
FILE_NAME="$1_$2"
python py_split_dna.py ../dna_large_file "$1" "$2" "$FILE_NAME"
./parse-examl -s "$FILE_NAME" -m DNA -n "$FILE_NAME".unpartitioned
./raxmlHPC-AVX -y -m GTRCAT -p 12345 -s "$FILE_NAME" -n tree
cp RAxML_parsimonyTree.tree "$FILE_NAME".tree
mv "$FILE_NAME".tree ../
mv "$FILE_NAME".unpartitioned.binary ../
mv "$FILE_NAME" ../
rm RAxML_*