#pragma once
#include <iostream>
#include <stdio.h>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h> 
#include <curand.h>
#include <cuda.h>
#include "activation.h"



namespace NeuronalNet 
{
	__host__ void GPU_CUDA_memcpyTest();
	__global__ void kernel_memcpyTest1(float* ptrA,float *ptrB, size_t count);
	__global__ void kernel_memcpyTest2(float* ptrA,float *ptrB, size_t count);


	// Interface functions
	

	__host__ void GPU_CUDA_getSpecs();
	__host__ void GPU_CUDA_calculateNet(float* weights, float* signals, float* outpuSignals,
										size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, Activation activation);

	__host__ void GPU_CUDA_getRandomWeight(float min, float max, float* h_list, size_t elements);

	__host__ void GPU_CUDA_allocMem(float* &d_list, size_t byteCount);
	__host__ void GPU_CUDA_freeMem(float* &d_list);

	__host__ void GPU_CUDA_transferToDevice(float* d_list, float* h_list, size_t byteCount);
	__host__ void GPU_CUDA_transferToHost(  float* d_list, float* h_list, size_t byteCount);


	// Kernel functions
	typedef float kernel_ActFp(float);
	__device__ inline float kernel_net_activation_linear(float x);
	__device__ inline float kernel_net_activation_gaussian(float x);
	__device__ inline float kernel_net_activation_sigmoid(float x);


	__device__ kernel_ActFp* kernel_net_getActivationFunction(Activation act);


	__global__ void kernel_net_calculateLayer(float* weights, float* inputSignals, float* outputSignals,
											  size_t neuronCount, size_t inputSignalCount, kernel_ActFp* act);
	__global__ void kernel_calculateNet(float* weights, float* signals, float* outpuSignals,
										size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, Activation act);

	//__global__ void kernel_scaleRandomWeight(float min, float max, float* d_list, size_t elements);


	__host__ inline void cuda_handleError(cudaError_t err)
	{
		switch (err)
		{
			case cudaError_t::cudaSuccess:
				return;
			default:
			{
				std::cout << "CudaError: " << err << "\n";
			}
		}
	}
}