
/*
__global__ static void
cudaOptimizedReduceKernel(double *g_idata, double *g_odata, unsigned int n)
{
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  if (BLOCK_SIZE >= 512)
  {
    if (tid < 256)
    {
      sdata[tid] += sdata[tid + 256];
    }
    __syncthreads();
  }
  if (BLOCK_SIZE >= 256)
  {
    if (tid < 128)
    {
      sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();
  }
  if (BLOCK_SIZE >= 128)
  {
    if (tid < 64)
    {
      sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();
  }
  if (tid < 32)
  {
    if (BLOCK_SIZE >= 64)
      sdata[tid] += sdata[tid + 32];
    if (BLOCK_SIZE >= 32)
      sdata[tid] += sdata[tid + 16];
    if (BLOCK_SIZE >= 16)
      sdata[tid] += sdata[tid + 8];
    if (BLOCK_SIZE >= 8)
      sdata[tid] += sdata[tid + 4];
    if (BLOCK_SIZE >= 4)
      sdata[tid] += sdata[tid + 2];
    if (BLOCK_SIZE >= 2)
      sdata[tid] += sdata[tid + 1];
  }
}

__global__ static void cudaEvaluateLeft(int *wptr, double *x2_start,
                                        double *tipVector, unsigned char *tipX1,
                                        double *diagptable, double *output,
                                        const int states, const int span,
                                        const int limit)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= limit)
  {
    output[i] = 0.0;
    return;
  }
  int j, k;
  double term;
  double *x1 = &(tipVector[states * tipX1[i]]);
  double *x2 = &(x2_start[span * i]);
  for (j = 0, term = 0.0; j < 4; j++)
    for (k = 0; k < states; k++)
      term += x1[k] * x2[j * states + k] * diagptable[j * states + k];
  output[i] = LOG(0.25 * FABS(term));
}

__global__ static void cudaEvaluateRight(int *wptr, double *x1_start,
                                         double *x2_start, double *diagptable,
                                         double *output, const int states,
                                         const int span, const int limit)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= limit)
  {
    output[i] = 0.0;
    return;
  }
  int j, k;
  double term;
  double *x1 = &(x1_start[span * i]);
  double *x2 = &(x2_start[span * i]);
  for (j = 0, term = 0.0; j < 4; j++)
    for (k = 0; k < states; k++)
      term +=
          x1[j * states + k] * x2[j * states + k] * diagptable[j * states + k];
  output[i] = LOG(0.25 * FABS(term));
}*/

int main(){
  double sum = 0.0;
  const int span = states * 4;
  if (tipX1) {
    cudaMemcpy(p->wgt, wptr, p->wgtSize, cudaMemcpyHostToDevice);
    cudaMemcpy(p->x2, x2_start, p->xSize, cudaMemcpyHostToDevice);
    cudaMemcpy(p->tipVector, tipVector, p->tipVectorSize,
               cudaMemcpyHostToDevice);
    cudaMemcpy(p->diagptable, diagptable, p->leftRightSize,
               cudaMemcpyHostToDevice);

    cudaEvaluateLeft<<<cudaBestGrid(n), BLOCK_SIZE>>>(
        p->wgt, p->x2, p->tipVector, tipX1, p->diagptable, p->outputBuffer,
        states, span, n);
  } else {
    cudaMemcpy(p->wgt, wptr, p->wgtSize, cudaMemcpyHostToDevice);
    cudaMemcpy(p->x1, x1_start, p->xSize, cudaMemcpyHostToDevice);
    cudaMemcpy(p->x2, x2_start, p->xSize, cudaMemcpyHostToDevice);
    cudaMemcpy(p->diagptable, diagptable, p->leftRightSize,
               cudaMemcpyHostToDevice);

    cudaEvaluateRight<<<cudaBestGrid(n), BLOCK_SIZE>>>(
        p->wgt, p->x1, p->x2, p->diagptable, p->outputBuffer, states, span, n);
  }
  cudaOptimizedReduceKernel<<<cudaBestGrid(n), BLOCK_SIZE>>>(p->outputBuffer,
                                                             p->dReduce, n);
  cudaMemcpy(p->hReduce, p->dReduce, p->gridSize * sizeof(double),
             cudaMemcpyDeviceToHost);
  int i = 0;
  for (i = 0; i < p->gridSize; i++) {
    sum += p->hReduce[i];
  }
  printf("SUM: %lf\n", sum);
  return sum;
}