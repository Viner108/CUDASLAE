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

	double EPS = 1.e-15;// точность приближенного решения 
	int N = 10240;//число уравнений в системе
	int size = N * N;//размер матрицы системы
	int N_thread = 512;//число нитей в блоке
	int N_blocks;//Число блоков
	unsigned int mem_sizeA = sizeof(double) * size;// память для матрицы
	unsigned int mem_sizeX = sizeof(double) * N;//память для столбцов
	// Выделение памяти на хост
	hA = (double*)malloc(mem_sizeA);// матрица А
	hF = (double*)malloc(mem_sizeX);//правая часть системы F
	hX = (double*)malloc(mem_sizeX);// точное решение
	hX0 = (double*)malloc(mem_sizeX);// приближенное решение X(n)
	hX1 = (double*)malloc(mem_sizeX);// приближенное решение X(n+1)
	hDelta = (double*)malloc(mem_sizeX);//разница |X(n+1) - X(n)|

	// Генерация матрицы A
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (i == j)
				hA[i * N + j] = 2.0; // Диагональные элементы
			else
				hA[i * N + j] = 1.0; // Внедиагональные элементы
		}
	}

	// Задание точного решения X и начального приближения X0
	for (int i = 0; i < N; i++) {
		hX[i] = 1.0; // Предположим, что точное решение состоит из единиц
		hX0[i] = 0.0; // Начальное приближение
	}

	// Задание правой части системы F
	for (int i = 0; i < N; i++) {
		double sum = 0.0;
		for (int j = 0; j < N; j++) {
			sum += hA[i * N + j] * hX[j];
		}
		hF[i] = sum;
	}
	// Выделение памяти на девайс
	cudaError_t error = cudaMalloc((void**)&dA, mem_sizeA);
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
		
	}// матрица А
	error = cudaMalloc((void**)&dF, mem_sizeX);
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
		
	}//правая часть системы F
	error = cudaMalloc((void**)&dX, mem_sizeX);
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
		
	}// точное решение
	error = cudaMalloc((void**)&dX0, mem_sizeX);
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
		
	}// приближенное решение X(n)
	error = cudaMalloc((void**)&dX1, mem_sizeX);
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
		
	}// приближенное решение X(n+1)
	error = cudaMalloc((void**)&delta, mem_sizeX);
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
		
	}//разница |X(n+1) - X(n)|

	N_blocks = (N + N_thread - 1) / N_thread; // задание сетки блоков

	// копирование данных с хост на девайс
	cudaMemcpy(dA, hA, mem_sizeA, cudaMemcpyHostToDevice);// матрица А
	cudaMemcpy(dF, hF, mem_sizeX, cudaMemcpyHostToDevice); // правая часть F
	cudaMemcpy(dX0, hX0, mem_sizeX, cudaMemcpyHostToDevice);// начальное приближение

	double eps1 = 1.;
	int k = 0;
	cudaEventRecord(start, 0);//Старт таймера
	while (eps1 > EPS) {//итерационный процесс
		k++;// номер итерации
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


	cudaEventRecord(start1, 0);//Старт таймера
	eps1 = 1.;
	k = 0;
	while (eps1 > EPS) {//итерационный процесс
		k++;// номер итерации
		for (int i = 0; i < N; i++) {
			double sum = 0.;
			for (int j = 0; j < N; j++) {
				sum += hA[j + i * N] * hX0[j];
			}
			hX1[i] = hX0[i] + (hF[i] - sum) / hA[i + i * N];
		}
		// Оценка точности решения
		double maxDelta = 0.0;
		for (int i = 0; i < N; i++) {
			double delta = fabs(hX1[i] - hX0[i]);
			if (delta > maxDelta) maxDelta = delta;
			hX0[i] = hX1[i]; // Обновление приближения
		}
		eps1 = maxDelta; // Максимальная разница между текущим и предыдущим приближениями
 	}
	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&timerValueCPU, start1, stop1);
	printf("\n CPU calculation time: %f ms\n", timerValueCPU);

}
