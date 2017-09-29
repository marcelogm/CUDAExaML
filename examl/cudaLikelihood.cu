#include "axml.h"    

__global__ static void cudaTipTipPrecomputeKernel(double *tipVector, double *left,
                                           double *right, double *umpX1,
                                           double *umpX2,
                                           const int maxStateValue,
                                           const int span, const int states) {

  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = n / maxStateValue, k = n % span;
  int l = 0;
  double *v = &(tipVector[states * i]);
  umpX1[n] = 0.0;
  umpX2[n] = 0.0;
  for (l = 0; l < states; l++) {
    umpX1[n] += v[l] * left[k * states + l];
    umpX2[n] += v[l] * right[k * states + l];
  }
}

__global__ static void cudaTipTipComputeKernel(double *x3, double *extEV,
                                        double *umpX1, double *umpX2,
                                        unsigned char *tipX1,
                                        unsigned char *tipX2, const int span,
                                        const int states, const int limit) {

  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= limit)
    return;
  double x1px2;
  const int i = n / 4, j = n % 4;
  double *uX1 = &umpX1[span * tipX1[i]];
  double *uX2 = &umpX2[span * tipX2[i]];
  double *v = &x3[i * span + j * states];
  int k, l;
  for (k = 0; k < states; k++)
    v[k] = 0.0;
  for (k = 0; k < states; k++) {
    x1px2 = uX1[j * states + k] * uX2[j * states + k];
    for (l = 0; l < states; l++)
      v[l] += x1px2 * extEV[states * k + l];
  }
}

__global__ static void cudaTipInnerPrecomputeKernel(double *tipVector, double *left,
                                             double *ump,
                                             const int maxStateValue,
                                             const int span, const int states) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = n / maxStateValue, k = n % span;
  int l = 0;
  double *v = &(tipVector[states * i]);
  ump[n] = 0.0;
  for (l = 0; l < states; l++) {
    ump[n] += v[l] * left[k * states + l];
  }
}

__global__ static void cudaTipInnerComputeKernel(double *x2, double *x3, double *extEV,
                                          unsigned char *tipX1,
                                          unsigned char *tipX2, double *right,
                                          double *umpX1, double *umpX2,
                                          const int span, const int states,
                                          const int limit) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= limit)
    return;
  int l, j;
  const int statesSquare = states * states, i = n / 4, k = n % 4;
  double x1px2, *v, *uX1, *ump_x2 = &(umpX2[states * n]);

  uX1 = &umpX1[span * tipX1[i]];
  v = &(x2[span * i + k * states]);
  for (l = 0; l < states; l++) {
    ump_x2[l] = 0.0;
    for (j = 0; j < states; j++)
      ump_x2[l] += v[j] * right[k * statesSquare + l * states + j];
  }
  v = &(x3[span * i + states * k]);
  for (l = 0; l < states; l++)
    v[l] = 0;
  for (l = 0; l < states; l++) {
    x1px2 = uX1[k * states + l] * ump_x2[l];
    for (j = 0; j < states; j++)
      v[j] += x1px2 * extEV[l * states + j];
  }
}

__global__ static void cudaInnerInnerComputeKernel(double *x1, double *x2, double *x3,
                                            double *extEV, double *left,
                                            double *right, const int span,
                                            const int states, const int limit) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= limit)
    return;
  int l, j;
  const int statesSquare = states * states, i = n / 4, k = n % 4;
  double x1px2, ar, al, *v, *vl, *vr;
  vl = &(x1[span * i + states * k]);
  vr = &(x2[span * i + states * k]);
  v = &(x3[span * i + states * k]);
  for (l = 0; l < states; l++)
    v[l] = 0;
  for (l = 0; l < states; l++) {
    al = 0.0;
    ar = 0.0;
    for (j = 0; j < states; j++) {
      al += vl[j] * left[k * statesSquare + l * states + j];
      ar += vr[j] * right[k * statesSquare + l * states + j];
    }
    x1px2 = al * ar;
    for (j = 0; j < states; j++)
      v[j] += x1px2 * extEV[states * l + j];
  }
}

extern "C" CudaGP *cudaGPMalloc(const int n, const int states, const int maxStateValue) {
  const int statesSquare = states * states, span = states * 4,
            precomputeLength = maxStateValue * span;

  CudaGP *p = (CudaGP *)malloc(sizeof(CudaGP));
  p->x1Size = sizeof(double) * n * 4 * states;
  p->x2Size = sizeof(double) * n * 4 * states;
  p->x3Size = sizeof(double) * n * 4 * states;
  p->extEVSize = sizeof(double) * statesSquare;
  p->tipVectorSize = sizeof(double) * span * states;
  p->tipXSize = sizeof(unsigned char) * n;
  p->leftRightSize = sizeof(double) * statesSquare * 4;
  p->umpXSize = sizeof(double) * precomputeLength;
  p->umpXLargeSize = (n * states * 4 < 256) ? sizeof(double) * precomputeLength
                                            : sizeof(double) * n * states * 4;
  p->wgtSize = sizeof(int) * n;

  cudaMalloc(&p->x1, p->x1Size);
  cudaMalloc(&p->x2, p->x2Size);
  cudaMalloc(&p->x3, p->x3Size);
  cudaMalloc(&p->extEV, p->extEVSize);
  cudaMalloc(&p->tipVector, p->tipVectorSize);
  cudaMalloc(&p->tipX1, p->tipXSize);
  cudaMalloc(&p->tipX2, p->tipXSize);
  cudaMalloc(&p->left, p->leftRightSize);
  cudaMalloc(&p->right, p->leftRightSize);
  cudaMalloc(&p->umpX1, p->umpXSize);
  cudaMalloc(&p->umpX2, p->umpXLargeSize);

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
  cudaFree(p->tipX1);
  cudaFree(p->tipX2);
  cudaFree(p->left);
  cudaFree(p->right);
  cudaFree(p->umpX1);
  cudaFree(p->umpX2);
  free(p);
}

int cudaBestGrid(int n) {
  return (n / BLOCK_SIZE) + ((n % BLOCK_SIZE == 0) ? 0 : 1);
}

extern "C" void cudaNewViewGAMMA(int tipCase, double *x1, double *x2, double *x3,
                             double *extEV, double *tipVector,
                             unsigned char *tipX1, unsigned char *tipX2, int n,
                             double *left, double *right, int *wgt,
                             int *scalerIncrement, CudaGP *p) {
  int i = 0, l, scale, addScale = 0;
  double *v;

  // Pode ser desnecessario
  cudaMemcpy(p->x3, x3, p->x3Size, cudaMemcpyHostToDevice);
  cudaMemcpy(p->extEV, extEV, p->extEVSize, cudaMemcpyHostToDevice);
  cudaMemcpy(p->left, left, p->leftRightSize, cudaMemcpyHostToDevice);
  cudaMemcpy(p->right, right, p->leftRightSize, cudaMemcpyHostToDevice);

  switch (tipCase) {
  case TIP_TIP: {
    cudaMemcpy(p->tipVector, tipVector, p->tipVectorSize,
               cudaMemcpyHostToDevice);
    cudaMemcpy(p->tipX1, tipX1, p->tipXSize, cudaMemcpyHostToDevice);
    cudaMemcpy(p->tipX2, tipX2, p->tipXSize, cudaMemcpyHostToDevice);
    cudaTipTipPrecomputeKernel<<<p->maxStateValue, p->span>>>(
        p->tipVector, p->left, p->right, p->umpX1, p->umpX2, p->maxStateValue,
        p->span, p->states);

    const int grid = cudaBestGrid(n * 4);
    cudaTipTipComputeKernel<<<grid, BLOCK_SIZE>>>(p->x3, p->extEV, p->umpX1,
                                                  p->umpX2, p->tipX1, p->tipX2,
                                                  p->span, p->states, n * 4);

    cudaMemcpy(x3, p->x3, p->x3Size, cudaMemcpyDeviceToHost);
  } break;
  case TIP_INNER: {
    cudaMemcpy(p->x2, x2, p->x2Size, cudaMemcpyHostToDevice);
    cudaMemcpy(p->tipVector, tipVector, p->tipVectorSize,
               cudaMemcpyHostToDevice);
    cudaMemcpy(p->tipX1, tipX1, p->tipXSize, cudaMemcpyHostToDevice);
    cudaMemcpy(p->tipX2, tipX2, p->tipXSize, cudaMemcpyHostToDevice);

    cudaTipInnerPrecomputeKernel<<<p->maxStateValue, p->span>>>(
        p->tipVector, p->left, p->umpX1, p->maxStateValue, p->span, p->states);

    const int grid = cudaBestGrid(n * 4);
    cudaTipInnerComputeKernel<<<grid, BLOCK_SIZE>>>(
        p->x2, p->x3, p->extEV, p->tipX1, p->tipX2, p->right, p->umpX1,
        p->umpX2, p->span, p->states, n * 4);
    cudaMemcpy(x3, p->x3, p->x3Size, cudaMemcpyDeviceToHost);

    for (i = 0; i < n; i++) {
      v = &x3[p->span * i];
      scale = 1;
      for (l = 0; scale && (l < p->span); l++) {
        scale = (ABS(v[l]) < minlikelihood);
      }
      if (scale) {
        for (l = 0; l < p->span; l++)
          v[l] *= twotothe256;
        addScale += wgt[i];
      }
    }
  } break;
  case INNER_INNER: {
    cudaMemcpy(p->x1, x1, p->x2Size, cudaMemcpyHostToDevice);
    cudaMemcpy(p->x2, x2, p->x2Size, cudaMemcpyHostToDevice);

    const int grid = cudaBestGrid(n * 4);
    cudaInnerInnerComputeKernel<<<grid, BLOCK_SIZE>>>(
        p->x1, p->x2, p->x3, p->extEV, p->left, p->right, p->span, p->states,
        n * 4);
    cudaMemcpy(x3, p->x3, p->x3Size, cudaMemcpyDeviceToHost);

    for (i = 0; i < n; i++) {
      v = &x3[p->span * i];
      scale = 1;
      for (l = 0; scale && (l < p->span); l++) {
        scale = (ABS(v[l]) < minlikelihood);
      }
      if (scale) {
        for (l = 0; l < p->span; l++)
          v[l] *= twotothe256;
        addScale += wgt[i];
      }
    }
  } break;
  default:
    assert(0);
  }

  *scalerIncrement = addScale;
}