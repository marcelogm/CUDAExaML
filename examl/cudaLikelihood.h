#include "axml.h"

extern void cudaMallocXVector(double **x, unsigned int size);

extern void cudaGPFillYVector(CudaGP *dst, unsigned char *src);

extern void cudaGPFillEV(CudaGP *dst, double *origin, unsigned int size);

extern void cudaGPFillTipVector(CudaGP *dst, double *origin, unsigned int size);

extern void cudaGPFillWgt(CudaGP *dst, int *origin, unsigned int size);

extern CudaGP *cudaGPMalloc(const int n, const int states,
                            const int maxStateValue, const int taxa);

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