#include "kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

__global__ void function(double* dA, double* dF, double* dX0, double* dX1, int N)
{
	double aa;
	double sum = 0.;
	int t = blockDim.x * blockIdx.x + threadIdx.x;

	for (int j = 0; j < N; j++) {
		sum += dA[j + t * N] * dX0[j];
		if (j == t) {
			aa = dA[j + t * N];
		}
		
	}
	dX1[t] = dX0[t] + (dF[t] - sum) / aa;
}

__global__ void eps(double* dX0, double* dX1, double* delta, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	delta[i] = abs(dX0[i] - dX1[i]);
	dX0[i] = dX1[i];
}
