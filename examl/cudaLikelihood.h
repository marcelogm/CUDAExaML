#include "axml.h"

extern void cudaGPAllocXVector(double **x, unsigned int size);

extern void cudaGPCopyModel(CudaGP *dst, double *evSrc, unsigned int evSize,
                            double *tipSrc, unsigned int tipSize);

extern CudaGP *cudaGPAlloc(const int n, const int states,
                           const int maxStateValue, const int taxa,
                           unsigned char *yResource, int *wgt);

extern void cudaNewViewGAMMA(int tipCase, double *x1, double *x2, double *x3,
                             unsigned char *tipX1, unsigned char *tipX2, int n,
                             double *left, double *right, int *wgt,
                             int *scalerIncrement, CudaGP *p);

extern double cudaEvaluateGAMMA(int *wptr, double *x1_start, double *x2_start,
                                unsigned char *tipX1, const int n,
                                double *diagptable, CudaGP *p);

extern void cudaSumGAMMA(int tipCase, double *sumtable, double *x1, double *x2,
                         unsigned char *tipX1, unsigned char *tipX2, int n,
                         CudaGP *p);