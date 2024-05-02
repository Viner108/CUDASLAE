#include "cuda_runtime.h"
#ifndef KERNEL_H
#define KERNEL_H


__global__ void function(double* dA, double* dF, double* dX0, double* dX1, int N);

__global__ void eps(double* dX0, double* dX1, double* delta, int N);

#endif // KERNEL_H