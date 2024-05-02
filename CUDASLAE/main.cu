#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernel.h"

int main()
{
	double* hA;
	double* hF;
	double* hX;
	double* hX0;
	double* hX1;
	double* hDelta;

	double* dA;
	double* dF;
	double* dX;
	double* dX0;
	double* dX1;
	double* delta;
	
	float timerValueGPU;
	float timerValueCPU;
	cudaEvent_t start, stop;
	cudaEvent_t start1, stop1;
	cudaEventCreate(&start);
	cudaEventCreate(&start1);
	cudaEventCreate(&stop);
	cudaEventCreate(&stop1);

	double EPS = 1.e-15;// �������� ������������� ������� 
	int N = 10240;//����� ��������� � �������
	int size = N * N;//������ ������� �������
	int N_thread = 512;//����� ����� � �����
	int N_blocks;//����� ������
	unsigned int mem_sizeA = sizeof(double) * size;// ������ ��� �������
	unsigned int mem_sizeX = sizeof(double) * N;//������ ��� ��������
	// ��������� ������ �� ����
	hA = (double*)malloc(mem_sizeA);// ������� �
	hF = (double*)malloc(mem_sizeX);//������ ����� ������� F
	hX = (double*)malloc(mem_sizeX);// ������ �������
	hX0 = (double*)malloc(mem_sizeX);// ������������ ������� X(n)
	hX1 = (double*)malloc(mem_sizeX);// ������������ ������� X(n+1)
	hDelta = (double*)malloc(mem_sizeX);//������� |X(n+1) - X(n)|

	// ��������� ������� A
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (i == j)
				hA[i * N + j] = 2.0; // ������������ ��������
			else
				hA[i * N + j] = 1.0; // ��������������� ��������
		}
	}

	// ������� ������� ������� X � ���������� ����������� X0
	for (int i = 0; i < N; i++) {
		hX[i] = 1.0; // �����������, ��� ������ ������� ������� �� ������
		hX0[i] = 0.0; // ��������� �����������
	}

	// ������� ������ ����� ������� F
	for (int i = 0; i < N; i++) {
		double sum = 0.0;
		for (int j = 0; j < N; j++) {
			sum += hA[i * N + j] * hX[j];
		}
		hF[i] = sum;
	}
	// ��������� ������ �� ������
	cudaError_t error = cudaMalloc((void**)&dA, mem_sizeA);
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
		
	}// ������� �
	error = cudaMalloc((void**)&dF, mem_sizeX);
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
		
	}//������ ����� ������� F
	error = cudaMalloc((void**)&dX, mem_sizeX);
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
		
	}// ������ �������
	error = cudaMalloc((void**)&dX0, mem_sizeX);
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
		
	}// ������������ ������� X(n)
	error = cudaMalloc((void**)&dX1, mem_sizeX);
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
		
	}// ������������ ������� X(n+1)
	error = cudaMalloc((void**)&delta, mem_sizeX);
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
		
	}//������� |X(n+1) - X(n)|

	N_blocks = (N + N_thread - 1) / N_thread; // ������� ����� ������

	// ����������� ������ � ���� �� ������
	cudaMemcpy(dA, hA, mem_sizeA, cudaMemcpyHostToDevice);// ������� �
	cudaMemcpy(dF, hF, mem_sizeX, cudaMemcpyHostToDevice); // ������ ����� F
	cudaMemcpy(dX0, hX0, mem_sizeX, cudaMemcpyHostToDevice);// ��������� �����������

	double eps1 = 1.;
	int k = 0;
	cudaEventRecord(start, 0);//����� �������
	while (eps1 > EPS) {//������������ �������
		k++;// ����� ��������
		function << <N_blocks, N_thread >> > (dA, dF, dX0, dX1, N);
		eps << <N_blocks, N_thread >> > (dX0, dX1, delta, N);
		cudaMemcpy(hDelta, delta, mem_sizeX, cudaMemcpyDeviceToHost);
		eps1 = 0.;
		for (int j = 0; j < N; j++)
		{
			eps1 += hDelta[j];
		}
		eps1 = eps1 / N;
		printf("\n Eps[%i] = %e ", k, eps);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueGPU, start, stop);
	printf("\n GPU calculation time: %f ms\n", timerValueGPU);

	cudaMemcpy(hX1, dX0, mem_sizeX, cudaMemcpyDeviceToHost);


	cudaEventRecord(start1, 0);//����� �������
	eps1 = 1.;
	k = 0;
	while (eps1 > EPS) {//������������ �������
		k++;// ����� ��������
		for (int i = 0; i < N; i++) {
			double sum = 0.;
			for (int j = 0; j < N; j++) {
				sum += hA[j + i * N] * hX0[j];
			}
			hX1[i] = hX0[i] + (hF[i] - sum) / hA[i + i * N];
		}
		// ������ �������� �������
		double maxDelta = 0.0;
		for (int i = 0; i < N; i++) {
			double delta = fabs(hX1[i] - hX0[i]);
			if (delta > maxDelta) maxDelta = delta;
			hX0[i] = hX1[i]; // ���������� �����������
		}
		eps1 = maxDelta; // ������������ ������� ����� ������� � ���������� �������������
 	}
	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&timerValueCPU, start1, stop1);
	printf("\n CPU calculation time: %f ms\n", timerValueCPU);

}
