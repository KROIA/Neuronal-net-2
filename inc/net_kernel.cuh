#pragma once
#include <iostream>
#include <stdio.h>


#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h> 
#include <curand.h>

#include <cuda.h>
#include <math.h>
#include <chrono>
#include "activation.h"




namespace NeuronalNet 
{
	// Interface functions
	
	__host__ double GPU_CUDA_transposeMatrix(float* d_list, size_t width);
	__host__ double GPU_CUDA_transposeMatrix2(float* d_list1,float* d_list2, size_t width);

	__host__ cudaDeviceProp GPU_CUDA_getSpecs();
	__host__ void GPU_CUDA_calculateNet(float* weights, float* signals, float* outpuSignals,
										size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, Activation activation);

	__host__ void GPU_CUDA_getRandomWeight(float min, float max, float* h_list, size_t elements);

	__host__ void GPU_CUDA_allocMem(float* &d_list, size_t byteCount);
	__host__ void GPU_CUDA_freeMem(float* &d_list);

	__host__ void GPU_CUDA_transferToDevice(float* d_list, float* h_list, size_t byteCount);
	__host__ void GPU_CUDA_transferToHost(  float* d_list, float* h_list, size_t byteCount);
	__host__ void GPU_CUDA_convertWeightToGPUWeight(float *d_list, size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount);


	__host__ size_t gaussSum(size_t val);
	__host__ size_t invGaussSum(size_t sum);

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

	__global__ void kernel_convertLayerWeightToGPUWeight(float* d_list, size_t signalCount, size_t neuronCount);
	__global__ void kernel_transposeMatrix(float* d_list, size_t width, size_t maxIndex, size_t indexOffset);
	__device__ inline size_t kernel_gaussSum(size_t val);
	__device__ inline size_t kernel_invGaussSum(size_t sum);
	__device__ inline void kernel_convertLayerWeightToGPUWeight_getNewIndex(size_t startIndex, size_t& endIndex, size_t signalCount, size_t neuronCount);

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