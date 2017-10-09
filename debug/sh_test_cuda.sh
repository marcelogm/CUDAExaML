#!/bin/bash
rm ExaML_*
echo "File $1:"
cd ../../ExaML/examl/
make -f Makefile.AVX.CUDA.gcc
rm *.o
mv examl-CUDA ../../Test/bin/examl-new
cd ../../Test/bin
echo "ExaML CUDA new:"
time ./examl-new -t ../"$1".tree -m GAMMA -s ../"$1".unpartitioned.binary -n T1 >> dump
rm ExaML_*
echo "ExaML CUDA:"
time ./examl-CUDA -t ../"$1".tree -m GAMMA -s ../"$1".unpartitioned.binary -n T1 >> dump
rm ExaML_*
