#pragma once
#include <iostream>
#include <stdio.h>



//#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <curand.h>


#include "config.h"
#include <cuda.h>
#include <math.h>
#include <chrono>
#include "activation.h"




namespace NeuronalNet 
{
	typedef enum 
	{
		toDevice = 0,
		toHost = 1
	} Direction;
	struct Point
	{
		size_t x;
		size_t y;
		size_t z;
	};
	struct CUDA_info
	{
		//size_t multiProcessorCount;
		//size_t maxBlocksPerMultiProcessor;
		size_t maxThreadsPerBlock;
		Point maxThreadDim;

		size_t totalGlobalMemory;
	};

	// Interface functions
	
	//__host__ double GPU_CUDA_transposeMatrix(float* d_list, size_t width);
	//__host__ double GPU_CUDA_transposeMatrix2(float* d_list1,float* d_list2, size_t width);
	NET_API __host__ void testCUDA();

	NET_API __host__ cudaDeviceProp GPU_CUDA_getSpecs();
	NET_API __host__ void GPU_CUDA_calculateNet(float* weights, float** multiSignalVec, float** multiOutputVec, size_t multiSignalSize,
										size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, Activation activation,
										CUDA_info * d_info = nullptr);

	NET_API __host__ void GPU_CUDA_getRandomWeight(float min, float max, float* h_list, size_t elements);

	
	template <typename T> 
	NET_API __host__  extern void GPU_CUDA_allocMem(T* &d_list, size_t byteCount);
	template <typename T>
	NET_API __host__ extern void GPU_CUDA_freeMem(T* &d_list);

	template <typename T>
	NET_API __host__ extern void GPU_CUDA_transferToDevice(T* d_list, T* h_list, size_t byteCount);
	template <typename T>
	NET_API __host__ extern void GPU_CUDA_transferToHost(  T* d_list, T* h_list, size_t byteCount);
	NET_API __host__ void GPU_CUDA_convertWeightMatrix(float *d_list, size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount,Direction dir);


	NET_API __host__ size_t gaussSum(size_t val);
	NET_API __host__ size_t invGaussSum(size_t sum);


	// Kernel global var.
	extern CUDA_info* _d_cudaInfo;
	extern CUDA_info* _h_cudaInfo;

	// Kernel functions
	typedef float kernel_ActFp(float);
	NET_API __device__ inline float kernel_net_activation_linear(float x);
	NET_API __device__ inline float kernel_net_activation_gaussian(float x);
	NET_API __device__ inline float kernel_net_activation_sigmoid(float x);


	NET_API __device__ kernel_ActFp* kernel_net_getActivationFunction(Activation act);


	NET_API __global__ void kernel_net_calculateLayer(float* weights, float* inputSignals, float* outputSignals,
											  size_t neuronCount, size_t inputSignalCount, kernel_ActFp* act);
	NET_API __global__ void kernel_calculateNet(float* weights, float** multiSignalVec, float** multiOutputVec, size_t multiSignalSize,
										size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, Activation act,
										CUDA_info* d_info = nullptr);

	//__global__ void kernel_convertLayerWeightToGPUWeight(float* d_list, size_t signalCount, size_t neuronCount);
	NET_API __global__ void kernel_transposeMatrix(float* d_list, size_t width, CUDA_info* d_info = nullptr);
	NET_API __global__ void kernel_transposeMatrix_internal(float* d_list, size_t width, size_t maxIndex, size_t indexOffset);
	NET_API __device__ inline size_t kernel_gaussSum(size_t val);
	NET_API __device__ inline size_t kernel_invGaussSum(size_t sum);

	NET_API __global__ void kernel_transposeMatrix(float* d_list, size_t width, size_t height, CUDA_info* d_info = nullptr);
	NET_API __global__ void kernel_transposeMatrix_rect_internal(float* d_list, float* tmpBuffer, size_t width, size_t height);
	

	

	NET_API __host__ void cuda_handleError(cudaError_t err);
}