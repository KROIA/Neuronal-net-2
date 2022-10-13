#pragma once
#include <iostream>
#include <stdio.h>



//#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <curand.h>


#include "config.h"
#include "debug.h"
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
	NET_API __host__ void GPU_CUDA_deleteSpecs();
	NET_API __host__ void GPU_CUDA_calculateNet(float* weights, float* biasList, float** multiSignalVec, float** multiOutputVec,
												float** multiConnectionSignalList, float** multiNetinputList, float** multiNeuronSignalList, size_t multiSignalSize,
												size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, Activation activation,
												CUDA_info* d_info = nullptr);

	NET_API __host__ void GPU_CUDA_getRandomValues(float* h_list, size_t elements, float min, float max);


	template <typename T>
	NET_API __host__  extern void GPU_CUDA_allocMem(T*& d_list, size_t byteCount);
	template <typename T>
	NET_API __host__ extern void GPU_CUDA_freeMem(T*& d_list);

	template <typename T>
	NET_API __host__ extern void GPU_CUDA_memset(T*& d_list, int value, size_t byteCount);
	template <typename T>
	NET_API __host__  extern void GPU_CUDA_memcpy(T*& d_source, T*& d_destination, size_t byteCount);

	template <typename T>
	NET_API __host__ extern void GPU_CUDA_transferToDevice(T* d_list, T* h_list, size_t byteCount);
	template <typename T>
	NET_API __host__ extern void GPU_CUDA_transferToHost(T* d_list, T* h_list, size_t byteCount);
	NET_API __host__ void GPU_CUDA_convertWeightMatrix(float* d_list, size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, Direction dir);


	NET_API __host__ void GPU_CUDA_learnBackpropagation(float* d_weights, float* d_deltaWeights, float* d_biasList, float* d_deltaBiasList, float* d_inputSignals, float* d_neuronOutputs, float* d_neuronNetinputs,
														size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, size_t neuronCount, size_t weightCount, Activation act,
														float* d_outputErrorList, float* d_expectedOutput, float learnParam);

	NET_API __host__ void GPU_CUDA_learnBackpropagationStream(float* d_weights, float** d_deltaWeights, float* d_biasList, float** d_deltaBiasList, float** d_inputSignals, float** d_neuronOutputs, float** d_neuronNetinputs,
															  size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, size_t neuronCount, size_t weightCount, Activation act,
															  float** d_outputErrorList, float** d_expectedOutput, float learnParam, size_t streamSize);
	NET_API __host__ void GPU_CUDA_learnBackpropagation_getOutputError(float* d_outputSignals, float* h_expectedOutputSignals,
																	   float* h_outputErrors, size_t outputCount);


	NET_API __host__ size_t gaussSum(size_t val);
	NET_API __host__ size_t invGaussSum(size_t sum);


	// Kernel global var.
	extern CUDA_info* _d_cudaInfo;
	extern CUDA_info* _h_cudaInfo;

	// Kernel functions
	typedef float kernel_ActFp(float);
	NET_API __device__ inline float kernel_net_activation_linear(float x);
	NET_API __device__ inline float kernel_net_activation_finiteLinear(float x);
	NET_API __device__ inline float kernel_net_activation_gaussian(float x);
	NET_API __device__ inline float kernel_net_activation_sigmoid(float x);
	NET_API __device__ inline float kernel_net_activation_binary(float x);

	NET_API __device__ inline float kernel_net_activation_linear_derivetive(float x);
	NET_API __device__ inline float kernel_net_activation_finiteLinear_derivetive(float x);
	NET_API __device__ inline float kernel_net_activation_gaussian_derivetive(float x);
	NET_API __device__ inline float kernel_net_activation_sigmoid_derivetive(float x);



	NET_API __device__ kernel_ActFp* kernel_net_getActivationFunction(Activation act);
	NET_API __device__ kernel_ActFp* kernel_net_getActivationDerivetiveFunction(Activation act);


	NET_API __global__ void kernel_net_calculateLayer(float* weights, float* biasList, float* inputSignals,
													  float* connectionSignalList, float* netinputList, float* neuronSignalList,
													  size_t neuronCount, size_t inputSignalCount, kernel_ActFp* act);
	NET_API __global__ void kernel_calculateNet(float* weights, float* biasList, float** multiSignalVec, float** multiOutputVec,
												float** multiConnectionSignalList, float** multiNetinputList, float** multiNeuronSignalList, size_t multiSignalSize,
												size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, Activation act,
												CUDA_info* d_info = nullptr);

	//__global__ void kernel_convertLayerWeightToGPUWeight(float* d_list, size_t signalCount, size_t neuronCount);
	NET_API __global__ void kernel_transposeMatrix(float* d_list, size_t width, CUDA_info* d_info = nullptr);
	NET_API __global__ void kernel_transposeMatrix_internal(float* d_list, size_t width, size_t maxIndex, size_t indexOffset);
	NET_API __device__ inline size_t kernel_gaussSum(size_t val);
	NET_API __device__ inline size_t kernel_invGaussSum(size_t sum);

	NET_API __global__ void kernel_transposeMatrix(float* d_list, size_t width, size_t height, CUDA_info* d_info = nullptr);
	NET_API __global__ void kernel_transposeMatrix_rect_internal(float* d_list, float* tmpBuffer, size_t width, size_t height);

	// Training algorithm

	NET_API __host__ void kernel_learnBackpropagationStream(float* d_weights, float** d_deltaWeights, float* d_biasList, float** d_deltaBiasList, float** d_inputSignals, float** d_neuronOutputs, float** d_neuronNetinputs,
														size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, size_t neuronCount, size_t weightCount, Activation act,
														float** d_outputErrorList, float** d_expectedOutput, float learnParam, size_t streamSize);

	NET_API __global__ void kernel_learnBackpropagation(float* d_weights, float **d_deltaWeights, float *d_biasList, float**d_deltaBiasList,
														float **d_inputSignals, float** d_neuronOutputs, float** d_neuronNetinputs,
														size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, size_t neuronCount, size_t weightCount, Activation act,//kernel_ActFp* actDerivPtr,
														float** d_outputErrorList, float** d_expectedOutput, size_t streamSize);

	NET_API __global__ void kernel_learnBackpropagation_applyDeltaValue(float* d_originalList, float** d_deltaList, float factor, size_t listSize, size_t cout);

	// Calculates the Error of the output layer
	NET_API __global__ void kernel_learnBackpropagation_getOutputError(float* d_outputSignals, float* d_expectedOutputSignals,
																	   float* d_outputErrors, size_t outputCount);

	// layerIY = y size of layer I,
	// layerJY = y size of layer I+1
	NET_API __global__ void kernel_learnBackpropagation_calculateLayerDeltaW(float* d_weights, float* d_deltaW, float *d_deltaB,
																			 float *d_neuronOutputs, float *d_netinputs,
																			 float *d_layerIErrorList, float* d_LayerJErrorList, 
																			 kernel_ActFp* actDerivPtr, size_t layerIY, size_t layerJY);

	//NET_API __device__ void kernel_learnBackpropagation_
	/*

	// Calculates the errorValue of the output neurons for each signalVector
	NET_API __device__ void kernel_calculateOutputError(float** d_netinpuitMultiSignals, float** d_outputMultiSignals, float** d_expectedOutputMultiSignal,
														float** d_errorMultiList, kernel_ActFp* derivetiveFunc,
														size_t outputNeuronStartIndex, size_t outputCount, size_t signalCount);
	// Calculates the errorValue of the hidden Layers
	NET_API __device__ void kernel_calculateHiddenError(float** d_netinpuitMultiSignals, float* d_weightList,
														float** d_errorMultiList, kernel_ActFp* derivetiveFunc,
														size_t hiddenNeuronStartIndex, size_t iNeuronYCount, 
														size_t jNeuronYCount, size_t signalCount);

	// Calcualtes the deltaW of each weight between the Layer I and J
	// The deltaW will also be applyied on the weightlist
	NET_API __device__ void kernel_changeLayerWeights(float* d_weightList, float** d_neuronMultiSignals, float** d_errorMultiList,
													  size_t neuronCountI, size_t neuronCountJ, size_t signalCount, float learnRate);
	// Calculates a slice of the function "kernel_changeLayerWeights"
	NET_API __device__ void kernel_changeLayerWeights_slice(float* d_deltaW, float** d_neuronMultiSignals, float** d_errorMultiList,
															size_t neuronCountI, size_t neuronCountJ, size_t signalCount, 
															size_t iteration,    size_t iterationSize);
	NET_API __device__ void kernel_applyDeltaWeight(float* d_deltaW, float* d_weights, size_t size);*/


	NET_API __global__ void kernel_offsetScale(float* d_list, float offset, float scale, size_t size, CUDA_info* d_info = nullptr);

	NET_API __host__ void cuda_handleError(cudaError_t err);
	NET_API __device__ void kernel_handleError(cudaError_t err);
};

