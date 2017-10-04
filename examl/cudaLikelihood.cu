#include "axml.h"

__global__ static void cudaTipTipPrecomputeKernel(
    double *tipVector, double *left, double *right, double *umpX1,
    double *umpX2, const int maxStateValue, const int span, const int states) {

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

__global__ static void
cudaTipTipComputeKernel(double *x3, double *extEV, double *umpX1, double *umpX2,
                        unsigned char *tipX1, unsigned char *tipX2,
                        const int span, const int states, const int limit) {

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

__global__ static void cudaTipInnerPrecomputeKernel(double *tipVector,
                                                    double *left, double *ump,
                                                    const int maxStateValue,
                                                    const int span,
                                                    const int states) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = n / maxStateValue, k = n % span;
  int l = 0;
  double *v = &(tipVector[states * i]);
  ump[n] = 0.0;
  for (l = 0; l < states; l++) {
    ump[n] += v[l] * left[k * states + l];
  }
}

__global__ static void
cudaTipInnerComputeKernel(double *x2, double *x3, double *extEV,
                          unsigned char *tipX1, unsigned char *tipX2,
                          double *right, double *umpX1, double *umpX2,
                          const int span, const int states, const int limit) {
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

__global__ static void
cudaInnerInnerComputeKernel(double *x1, double *x2, double *x3, double *extEV,
                            double *left, double *right, const int span,
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

__global__ static void cudaAtomicScale(double *x3, int *addScale, int *wgt,
                                       int span, int limit) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= limit)
    return;
  double *v = &x3[span * i];
  int l, scale = 1;
  for (l = 0; scale && (l < span); l++) {
    scale = (ABS(v[l]) < minlikelihood);
  }
  if (scale) {
    for (l = 0; l < span; l++)
      v[l] *= twotothe256;
    atomicAdd(addScale,wgt[i]);
  }
}

extern "C" void cudaGPFillVector(CudaGP *dst, unsigned char *src) {
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

static inline int cudaBestGrid(int n) {
  return (n / BLOCK_SIZE) + ((n % BLOCK_SIZE == 0) ? 0 : 1);
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
    cudaTipTipPrecomputeKernel<<<p->maxStateValue, p->span>>>(
        p->tipVector, p->left, p->right, p->umpX1, p->umpX2, p->maxStateValue,
        p->span, p->states);
    cudaTipTipComputeKernel<<<cudaBestGrid(n * 4), BLOCK_SIZE>>>(
        p->x3, p->extEV, p->umpX1, p->umpX2, tipX1, tipX2, p->span, p->states,
        n * 4);

    cudaMemcpy(x3, p->x3, p->xSize, cudaMemcpyDeviceToHost);
  } break;
  case TIP_INNER: {
    cudaMemcpy(p->x2, x2, p->xSize, cudaMemcpyHostToDevice);
    cudaMemcpy(p->tipVector, tipVector, p->tipVectorSize,
               cudaMemcpyHostToDevice);
    cudaMemcpy(p->wgt, wgt, p->wgtSize, cudaMemcpyHostToDevice);

    cudaTipInnerPrecomputeKernel<<<p->maxStateValue, p->span>>>(
        p->tipVector, p->left, p->umpX1, p->maxStateValue, p->span, p->states);

    cudaTipInnerComputeKernel<<<cudaBestGrid(n * 4), BLOCK_SIZE>>>(
        p->x2, p->x3, p->extEV, tipX1, tipX2, p->right, p->umpX1, p->umpX2,
        p->span, p->states, n * 4);

    cudaAtomicScale<<<cudaBestGrid(n), BLOCK_SIZE>>>(p->x3, p->addScale, p->wgt,
                                                     p->span, n);
    cudaMemcpy(x3, p->x3, p->xSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(&addScale, p->addScale, sizeof(int), cudaMemcpyDeviceToHost);
  } break;
  case INNER_INNER: {
    cudaMemcpy(p->x1, x1, p->xSize, cudaMemcpyHostToDevice);
    cudaMemcpy(p->x2, x2, p->xSize, cudaMemcpyHostToDevice);
    cudaMemcpy(p->wgt, wgt, p->wgtSize, cudaMemcpyHostToDevice);

    cudaInnerInnerComputeKernel<<<cudaBestGrid(n * 4), BLOCK_SIZE>>>(
        p->x1, p->x2, p->x3, p->extEV, p->left, p->right, p->span, p->states,
        n * 4);
    cudaAtomicScale<<<cudaBestGrid(n), BLOCK_SIZE>>>(p->x3, p->addScale, p->wgt,
                                                     p->span, n);
    cudaMemcpy(x3, p->x3, p->xSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(&addScale, p->addScale, sizeof(int), cudaMemcpyDeviceToHost);
  } break;
  default:
    assert(0);
  }

  *scalerIncrement = addScale;
}
