#!/bin/bash
rm ExaML_*
echo "File $1:"
echo "ExaML SSE:"
time ./examl -t ../"$1".tree -m GAMMA -s ../"$1".unpartitioned.binary -n T1 >> dump
rm ExaML_*
echo "ExaML AVX:"
time ./examl-AVX -t ../"$1".tree -m GAMMA -s ../"$1".unpartitioned.binary -n T1 >> dump
rm ExaML_*
echo "ExaML CUDA new:"
time ./examl-new -t ../"$1".tree -m GAMMA -s ../"$1".unpartitioned.binary -n T1 >> dump
rm ExaML_*
