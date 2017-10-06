#include "axml.h"
#include <math.h>

static inline int cudaBestGrid(int n) {
  return (n / BLOCK_SIZE) + ((n % BLOCK_SIZE == 0) ? 0 : 1);
}

__global__ static void cudaPreTTGammaKernel(double *v, double *l, double *r,
                                            double *umpX1, double *umpX2,
                                            const int maxStateValue) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, n = x / 2;
  v += 4 * (n / maxStateValue);
  if (x % 2) {
    l += (n % 16) * 4;
    umpX1[n] = v[0] * l[0] + v[1] * l[1] + v[2] * l[2] + v[3] * l[3];
  } else {
    r += (n % 16) * 4;
    umpX2[n] = v[0] * r[0] + v[1] * r[1] + v[2] * r[2] + v[3] * r[3];
  }
}

__global__ static void cudaTTGammaKernel(double *v, double *extEV, double *uX1,
                                         double *uX2, unsigned char *tipX1,
                                         unsigned char *tipX2,
                                         const int limit) {

  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= limit)
    return;
  const int i = n / 4, j = (n % 4) * 4;
  v += i * 16 + j;
  uX1 += 16 * tipX1[i] + j;
  uX2 += 16 * tipX2[i] + j;
  v[0] = v[1] = v[2] = v[3] = 0.0;
  double x1px2;
  for (int k = 0; k < 4; k++) {
    x1px2 = uX1[k] * uX2[k];
    v[0] += x1px2 * extEV[0];
    v[1] += x1px2 * extEV[1];
    v[2] += x1px2 * extEV[2];
    v[3] += x1px2 * extEV[3];
    extEV += 4;
  }
}

__global__ static void cudaPreTIPreGammaKernel(double *t, double *l,
                                               double *ump,
                                               const int maxStateValue) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  t += (n / maxStateValue) * 4;
  l += (n % 16) * 4;
  ump[n] = t[0] * l[0] + t[1] * l[1] + t[2] * l[2] + t[3] * l[3];
}

__global__ static void cudaTIGammaKernel(double *x2, double *x3, double *extEV,
                                         unsigned char *tipX1,
                                         unsigned char *tipX2, double *r,
                                         double *uX1, double *uX2,
                                         const int limit) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= limit)
    return;
  int l = 0;
  const int i = n / 4, k = n % 4;
  double x1px2;

  uX2 += 4 * n;
  uX1 += 16 * tipX1[i];
  r += k * 16;
  x2 += 16 * i + k * 4;

  for (l = 0; l < 4; l++) {
    uX2[l] = 0.0;
    uX2[l] += x2[0] * r[0];
    uX2[l] += x2[1] * r[1];
    uX2[l] += x2[2] * r[2];
    uX2[l] += x2[3] * r[3];
    r += 4;
  }

  x3 += 16 * i + 4 * k;
  x3[0] = x3[1] = x3[2] = x3[3] = 0.0;

  for (l = 0; l < 4; l++) {
    x1px2 = uX1[k * 4 + l] * uX2[l];
    x3[0] += x1px2 * extEV[0];
    x3[1] += x1px2 * extEV[1];
    x3[2] += x1px2 * extEV[2];
    x3[3] += x1px2 * extEV[3];
    extEV += 4;
  }
}

__global__ static void cudaIIGammaKernel(double *x1, double *x2, double *x3,
                                         double *extEV, double *left,
                                         double *right, const int limit) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= limit)
    return;
  int l;
  const int i = n / 4, k = n % 4;
  double x1px2, ar, al;
  const int offset = 16 * i + 4 * k;
  x1 += offset;
  x2 += offset;
  x3 += offset;

  x3[0] = x3[1] = x3[2] = x3[3] = 0.0;

  left += k * 16;
  right += k * 16;
  for (l = 0; l < 4; l++) {
    al = x1[0] * left[0] + x1[1] * left[1] + x1[2] * left[2] + x1[3] * left[3];
    ar = x2[0] * right[0] + x2[1] * right[1] + x2[2] * right[2] +
         x2[3] * right[3];
    left += 4;
    right += 4;

    x1px2 = al * ar;

    x3[0] += x1px2 * extEV[0];
    x3[1] += x1px2 * extEV[1];
    x3[2] += x1px2 * extEV[2];
    x3[3] += x1px2 * extEV[3];
    extEV += 4;
  }
}

__global__ static void cudaAScaleGammaKernel(double *x3, int *addScale,
                                             int *wgt, int limit) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= limit)
    return;
  x3 += 16 * i;
  int l, scale = 1;
  for (l = 0; scale && (l < 16); l++) {
    scale = (ABS(x3[l]) < minlikelihood);
  }
  if (scale) {
    for (l = 0; l < 16; l++)
      x3[l] *= twotothe256;
    atomicAdd(addScale, wgt[i]);
  }
}

__global__ static void cudaEvaluateLeftGamma(int *wptr, double *x2,
                                             double *tipVector,
                                             unsigned char *tipX1,
                                             double *diagptable, double *output,
                                             const int limit) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= limit) {
    output[i] = 0.0;
    return;
  }
  int j;
  double term = 0.0;
  tipVector += 4 * tipX1[i];
  x2 += 16 * i;
  for (j = 0; j < 4; j++) {
    term += tipVector[0] * x2[0] * diagptable[0] +
            tipVector[1] * x2[1] * diagptable[1] +
            tipVector[2] * x2[2] * diagptable[2] +
            tipVector[3] * x2[3] * diagptable[3];
    x2 += 4;
    diagptable += 4;
  }
  term = log(0.25 * fabs(term));
  output[i] = wptr[i] * term;
}

__global__ static void cudaEvaluateRightGamma(int *wptr, double *x1, double *x2,
                                              double *diagptable,
                                              double *output, const int limit) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= limit) {
    output[i] = 0.0;
    return;
  }
  int j;
  double term = 0.0;
  x1 += 16 * i;
  x2 += 16 * i;
  for (j = 0; j < 4; j++) {
    term += x1[0] * x2[0] * diagptable[0] + x1[1] * x2[1] * diagptable[1] +
            x1[2] * x2[2] * diagptable[2] + x1[3] * x2[3] * diagptable[3];
    x1 += 4;
    x2 += 4;
    diagptable += 4;
  }
  term = log(0.25 * fabs(term));
  output[i] = wptr[i] * term;
}


extern "C" void cudaGPFillYVector(CudaGP *dst, unsigned char *src) {
  cudaMemcpy(dst->yResource, src,
             dst->taxa * dst->length * sizeof(unsigned char),
             cudaMemcpyHostToDevice);
  int i = 0;
  dst->yVector =
      (unsigned char **)calloc(dst->taxa + 1, sizeof(unsigned char *));
  for (i = 1; i <= dst->taxa; ++i) {
    dst->yVector[i] = dst->yResource + (i - 1) * dst->length;
  }
}

extern "C" void cudaGPFillXVector(CudaGP *dst, unsigned int tips,
                                  long unsigned int size) {
  cudaMalloc(&dst->xResource, tips * size);
  cudaMemset(dst->xResource, 0, tips * size);
  int i = 0;
  dst->xVector = (double **)calloc(dst->taxa + 1, sizeof(double *));
  for (i = 1; i <= tips; ++i) {
    dst->xVector[i] = dst->xResource + (i - 1) * size;
  }
}

extern "C" CudaGP *cudaGPMalloc(const int n, const int states,
                                const int maxStateValue, const int taxa) {
  const int statesSquare = states * states, span = states * 4,
            precomputeLength = maxStateValue * span;

  CudaGP *p = (CudaGP *)malloc(sizeof(CudaGP));
  p->xSize = sizeof(double) * n * 4 * states;
  p->extEVSize = sizeof(double) * statesSquare;
  p->tipVectorSize = sizeof(double) * span * states;
  p->tipXSize = sizeof(unsigned char) * n;
  p->leftRightSize = sizeof(double) * statesSquare * 4;
  p->umpXSize = sizeof(double) * precomputeLength;
  p->umpXLargeSize = (n * states * 4 < 256) ? sizeof(double) * precomputeLength
                                            : sizeof(double) * n * states * 4;
  p->wgtSize = sizeof(int) * n;
  p->gridSize = cudaBestGrid(n);

  cudaMalloc(&p->wgt, p->wgtSize);
  cudaMalloc(&p->addScale, sizeof(int));

  cudaMalloc(&p->x1, p->xSize);
  cudaMalloc(&p->x2, p->xSize);
  cudaMalloc(&p->x3, p->xSize);
  cudaMalloc(&p->extEV, p->extEVSize);
  cudaMalloc(&p->tipVector, p->tipVectorSize);
  cudaMalloc(&p->left, p->leftRightSize);
  cudaMalloc(&p->right, p->leftRightSize);
  cudaMalloc(&p->umpX1, p->umpXSize);
  cudaMalloc(&p->umpX2, p->umpXLargeSize);
  cudaMalloc(&p->yResource, taxa * n * sizeof(unsigned char));
  cudaMalloc(&p->evaluateSum, sizeof(double));
  cudaMalloc(&p->diagptable, p->leftRightSize);

  cudaMalloc(&p->outputBuffer, p->gridSize * BLOCK_SIZE * sizeof(double));
  cudaMalloc(&p->dReduce, p->gridSize * sizeof(double));
  p->hReduce = (double *)malloc(sizeof(double) * p->gridSize);

  cudaMemset(p->wgt, 0, p->wgtSize);
  cudaMemset(p->addScale, 0, sizeof(int));

  p->length = n;
  p->taxa = taxa;
  p->states = states;
  p->statesSquare = states * states;
  p->span = states * 4;
  p->precomputeLength = maxStateValue * span;
  p->maxStateValue = maxStateValue;
  return p;
}

extern "C" void cudaGPFree(CudaGP *p) {
  cudaFree(p->x1);
  cudaFree(p->x2);
  cudaFree(p->x3);
  cudaFree(p->extEV);
  cudaFree(p->tipVector);
  cudaFree(p->left);
  cudaFree(p->right);
  cudaFree(p->umpX1);
  cudaFree(p->umpX2);
  free(p);
}

extern "C" double cudaEvaluateGAMMA(int *wptr, double *x1_start,
                                    double *x2_start, double *tipVector,
                                    unsigned char *tipX1, const int n,
                                    double *diagptable, const int states,
                                    CudaGP *p) {
  double sum = 0.0;
  int i;
  if (tipX1) {
    cudaMemcpy(p->wgt, wptr, p->wgtSize, cudaMemcpyHostToDevice);
    cudaMemcpy(p->x2, x2_start, p->xSize, cudaMemcpyHostToDevice);
    cudaMemcpy(p->tipVector, tipVector, p->tipVectorSize,
               cudaMemcpyHostToDevice);
    cudaMemcpy(p->diagptable, diagptable, p->leftRightSize,
               cudaMemcpyHostToDevice);

    cudaEvaluateLeftGamma<<<cudaBestGrid(n), BLOCK_SIZE>>>(
        p->wgt, p->x2, p->tipVector, tipX1, p->diagptable, p->outputBuffer, n);

    double *out = (double *)malloc(sizeof(double) * n);
    cudaMemcpy(out, p->outputBuffer, sizeof(double) * n,
               cudaMemcpyDeviceToHost);
    for (i = 0; i < n; i++) {
      sum += out[i];
    }
    free(out);
  } else {
    cudaMemcpy(p->wgt, wptr, p->wgtSize, cudaMemcpyHostToDevice);
    cudaMemcpy(p->x1, x1_start, p->xSize, cudaMemcpyHostToDevice);
    cudaMemcpy(p->x2, x2_start, p->xSize, cudaMemcpyHostToDevice);
    cudaMemcpy(p->diagptable, diagptable, p->leftRightSize,
               cudaMemcpyHostToDevice);

    cudaEvaluateRightGamma<<<cudaBestGrid(n), BLOCK_SIZE>>>(
        p->wgt, p->x1, p->x2, p->diagptable, p->outputBuffer, n);

    double *out = (double *)malloc(sizeof(double) * n);
    cudaMemcpy(out, p->outputBuffer, sizeof(double) * n,
               cudaMemcpyDeviceToHost);
    for (i = 0; i < n; i++) {
      sum += out[i];
    }
    free(out);
  }
  return sum;
}

extern "C" void cudaNewViewGAMMA(int tipCase, double *x1, double *x2,
                                 double *x3, double *extEV, double *tipVector,
                                 unsigned char *tipX1, unsigned char *tipX2,
                                 int n, double *left, double *right, int *wgt,
                                 int *scalerIncrement, CudaGP *p) {
  int addScale = 0;
  cudaMemcpy(p->extEV, extEV, p->extEVSize, cudaMemcpyHostToDevice);
  cudaMemcpy(p->left, left, p->leftRightSize, cudaMemcpyHostToDevice);
  cudaMemcpy(p->right, right, p->leftRightSize, cudaMemcpyHostToDevice);

  switch (tipCase) {
  case TIP_TIP: {
    cudaMemcpy(p->tipVector, tipVector, p->tipVectorSize,
               cudaMemcpyHostToDevice);
    cudaPreTTGammaKernel<<<p->maxStateValue * 2, p->span>>>(
        p->tipVector, p->left, p->right, p->umpX1, p->umpX2, p->maxStateValue);

    cudaTTGammaKernel<<<cudaBestGrid(n * 4), BLOCK_SIZE>>>(
        p->x3, p->extEV, p->umpX1, p->umpX2, tipX1, tipX2, n * 4);
    cudaMemcpy(x3, p->x3, p->xSize, cudaMemcpyDeviceToHost);
  } break;
  case TIP_INNER: {
    cudaMemcpy(p->x2, x2, p->xSize, cudaMemcpyHostToDevice);
    cudaMemcpy(p->tipVector, tipVector, p->tipVectorSize,
               cudaMemcpyHostToDevice);
    cudaMemcpy(p->wgt, wgt, p->wgtSize, cudaMemcpyHostToDevice);

    cudaPreTIPreGammaKernel<<<p->maxStateValue, p->span>>>(
        p->tipVector, p->left, p->umpX1, p->maxStateValue);

    cudaTIGammaKernel<<<cudaBestGrid(n * 4), BLOCK_SIZE>>>(
        p->x2, p->x3, p->extEV, tipX1, tipX2, p->right, p->umpX1, p->umpX2,
        n * 4);

    cudaAScaleGammaKernel<<<cudaBestGrid(n), BLOCK_SIZE>>>(p->x3, p->addScale,
                                                           p->wgt, n);
    cudaMemcpy(x3, p->x3, p->xSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(&addScale, p->addScale, sizeof(int), cudaMemcpyDeviceToHost);
  } break;
  case INNER_INNER: {
    cudaMemcpy(p->x1, x1, p->xSize, cudaMemcpyHostToDevice);
    cudaMemcpy(p->x2, x2, p->xSize, cudaMemcpyHostToDevice);
    cudaMemcpy(p->wgt, wgt, p->wgtSize, cudaMemcpyHostToDevice);

    cudaIIGammaKernel<<<cudaBestGrid(n * 4), BLOCK_SIZE>>>(
        p->x1, p->x2, p->x3, p->extEV, p->left, p->right, n * 4);
    cudaAScaleGammaKernel<<<cudaBestGrid(n), BLOCK_SIZE>>>(p->x3, p->addScale,
                                                           p->wgt, n);
    cudaMemcpy(x3, p->x3, p->xSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(&addScale, p->addScale, sizeof(int), cudaMemcpyDeviceToHost);
  } break;
  default:
    assert(0);
  }

  *scalerIncrement = addScale;
}