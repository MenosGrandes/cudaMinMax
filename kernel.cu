__global__ void seq_minmaxKernel(double* max, double* min, const double* a)
{
    __shared__ double maxtile[BLOCKSIZE];
    __shared__ double mintile[BLOCKSIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    maxtile[tid] = a[i];
    mintile[tid] = a[i];
    __syncthreads();

    //sequential addressing by reverse loop and thread-id based indexing
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (maxtile[tid + s] > maxtile[tid])
            {
                maxtile[tid] = maxtile[tid + s];
            }

            if (mintile[tid + s] < mintile[tid])
            {
                mintile[tid] = mintile[tid + s];
            }
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        max[blockIdx.x] = maxtile[0];
        min[blockIdx.x] = mintile[0];
    }
}
__global__ void seq_finalminmaxKernel(double* max, double* min)
{
    __shared__ double maxtile[BLOCKSIZE];
    __shared__ double mintile[BLOCKSIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    maxtile[tid] = max[i];
    mintile[tid] = min[i];
    __syncthreads();

    //sequential addressing by reverse loop and thread-id based indexing
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (maxtile[tid + s] > maxtile[tid])
            {
                maxtile[tid] = maxtile[tid + s];
            }

            if (mintile[tid + s] < mintile[tid])
            {
                mintile[tid] = mintile[tid + s];
            }
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        max[blockIdx.x] = maxtile[0];
        min[blockIdx.x] = mintile[0];
    }
}

