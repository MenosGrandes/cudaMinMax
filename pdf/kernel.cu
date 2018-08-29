__global__ void seq_minmaxKernel(double* max, double* min, const double* a)
{
    __shared__ double s_max[BLOCKSIZE];
    __shared__ double s_min[BLOCKSIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    s_max[tid] = s_min[tid] = a[i];
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (s_max[tid + s] > s_max[tid])
            {
                s_max[tid] = s_max[tid + s];
            }

            if (s_min[tid + s] < s_min[tid])
            {
                s_min[tid] = s_min[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        max[blockIdx.x] = s_max[0];
        min[blockIdx.x] = s_min[0];
    }

}
__global__ void seq_finalminmaxKernel(double* max, double* min)
{
    __shared__ double s_max[BLOCKSIZE];
    __shared__ double s_min[BLOCKSIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    s_max[tid] = max[i];
    s_min[tid] = min[i];
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (s_max[tid + s] > s_max[tid])
            {
                s_max[tid] = s_max[tid + s];
            }

            if (s_min[tid + s] < s_min[tid])
            {
                s_min[tid] = s_min[tid + s];
            }
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        max[blockIdx.x] = s_max[0];
        min[blockIdx.x] = s_min[0];
    }
}
