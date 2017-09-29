#include "axml.h"

extern CudaGP *cudaGPMalloc(const int n, const int states,
                            const int maxStateValue);

extern void cudaGPFree(CudaGP *p);

extern void cudaNewViewGAMMA(int tipCase, double *x1, double *x2, double *x3,
                             double *extEV, double *tipVector,
                             unsigned char *tipX1, unsigned char *tipX2, int n,
                             double *left, double *right, int *wgt,
                             int *scalerIncrement, CudaGP *p);