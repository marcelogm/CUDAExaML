#!/bin/bash
rm timer.log
touch timer.log
cd ../../ExaML/examl/
make -f Makefile.AVX.gcc
rm *.o
cp examl-AVX ../../Test/bin/examl-AVX
cd ../../Test/bin
./examl-AVX -t ../"$1".tree -m GAMMA -s ../"$1".unpartitioned.binary -n T1 
rm ExaML_*
