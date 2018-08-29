#ifndef __CUDACC__
    #define __CUDACC__
#endif
#include <math.h>
#include <stdio.h>
#include <random>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <random>
#include "random.hpp"
#include "helpers.cu"
#include "kernel.cu"
int  minmaxCuda(double* max, double* min, const double* a)
{
    double* dev_a = 0;
    double* dev_max = 0;
    double* dev_min = 0;
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid(SIZE);

    if (
        (cudaSetDevice(0) != cudaSuccess) ||
        (cudaMalloc((void**)&dev_max, SIZE * sizeof(double)) != cudaSuccess) ||
        (cudaMalloc((void**)&dev_min, SIZE * sizeof(double)) != cudaSuccess) ||
        (cudaMalloc((void**)&dev_a, SIZE * SIZE  * sizeof(double)) != cudaSuccess) ||
        (cudaMemcpy(dev_a, a, SIZE * SIZE * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
    )
    {
        fprintf(stderr, "cudaSetDevice failed/ cudaMalloc failed/cudaMemcpy failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    TIMERSTART(SeqMinMaxKernel);
    seq_minmaxKernel <<< dimGrid, dimBlock>>>(dev_max, dev_min, dev_a);
    cudaThreadSynchronize();
    seq_finalminmaxKernel <<< 1, dimBlock>>>(dev_max, dev_min);
    cudaThreadSynchronize();
    TIMERSTOP(SeqMinMaxKernel);
    CUERR;

    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code  after launching minmaxKernel!\n");
        goto Error;
    }

    if ((cudaMemcpy(max, dev_max, SIZE * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) ||
        (cudaMemcpy(min, dev_min, SIZE * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
       )
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    return 0;
Error:
    cudaFree(dev_max);
    cudaFree(dev_min);
    cudaFree(dev_a);
    return -1;
}

int main()
{
    Random < double, SIZE, -1, 1 > r;
    double* a = r.generateRandomArray();
    double* max = new double[SIZE];
    double* min = new double[SIZE];

    if (minmaxCuda(max, min, a))
    {
        fprintf(stderr, "minmaxCuda failed!");
        return 1;
    }

#ifdef DEBUG
    std::pair<float, float> minMax;

    for (int i = 0; i < SIZE * SIZE; i++)
    {
        if (a[i] < minMax.first)
        {
            minMax.first = a[i];
        }
        else if (a[i] > minMax.second)
        {
            minMax.second = a[i];
        }

        //std::cout << a[i] << " ";
    }

    std::cout << std::endl;
    std::cout << "CPU MIN: " << minMax.first << " MAX: " << minMax.second << std::endl;
    std::cout << "GPU MIN: " << min[0] << " MAX: " << max[0] << std::endl;
#endif

    if (cudaDeviceReset() != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        delete []a;
        delete []min;
        delete []max;
        return 1;
    }

    delete []a;
    delete []min;
    delete []max;
    return 0;
}


