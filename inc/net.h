#pragma once

#include <iostream>
#include <stdio.h>
#include <vector>

#include "layer.h"
#include "activation.h"
#include "net_kernel.cuh"


#include <chrono>


#define NET_DEBUG


// helper functions for cleaner time measuring code
extern std::chrono::time_point<std::chrono::high_resolution_clock> now();
template <typename T>
extern double milliseconds(T t);

enum Hardware
{
	cpu,
	gpu_cuda
};

typedef std::vector<float> SignalVector;




class Net
{
	typedef float ActFp(float);
	public:
	Net();
	~Net();

	void setDimensions(size_t inputs, size_t hiddenX, size_t hiddenY, size_t outputs);
	size_t getInputCount() const;
	size_t getHiddenXCount() const;
	size_t getHiddenYCount() const;
	size_t getOutputCount() const;

	void setActivation(Activation act);
	Activation getActivation() const;

	void setHardware(Hardware ware);
	Hardware getHardware() const;

	bool build();
	void randomizeWeights();
	bool randomizeWeights(size_t from, size_t to);
	static float getRandomValue(float min, float max);

	void setInputVector(float* signalList);
	void setInputVector(const SignalVector& signalList);
	void setInput(size_t index, float signal);
	float getInput(size_t index) const;
	SignalVector getInputVector() const;
	SignalVector getOutputVector() const;

	void calculate();

	protected:

	void CPU_calculate();
	void GPU_CUDA_calculate();
	static void CPU_calculateNet(float* weights, float* signals, float* outpuSignals,
							 size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, ActFp *activation);
	static void CPU_calculateLayer(float* weights, float* inputSignals, float* outputSignals,
							   size_t neuronCount, size_t inputSignalCount, ActFp* activation);

	void transferWeightsToDevice();
	void transferWeightsToHost();
	void transferSignalsToDevice();
	void transferSignalsToHost();
	void buildDevice();
	void destroyDevice();
	void buildHostWeights();
	void destroyHostWeights();

	size_t m_inputs;
	size_t m_hiddenX;
	size_t m_hiddenY;
	size_t m_outputs;

	size_t m_neuronCount;
	size_t m_weightsCount;

	Activation m_activation;
	ActFp* m_activationFunc;

	float* m_inputSignalList;
	float* m_weightsList;
	float* m_outputSingalList;
	bool   m_built;

	// Extern hardware
	Hardware m_hardware;
	float* d_inputSignalList;
	float* d_weightsList;
	float* d_outputSingalList;

	private:
	static inline float activation_linear(float inp);
	static inline float activation_gauss(float inp);
	static inline float activation_sigmoid(float inp);

};

#ifdef NET_DEBUG
#define CONSOLE(x) std::cout<<__FUNCTION__<<" : "<< x << "\n";
#else
#define CONSOLE(x)
#endif


#define __VERIFY_RANGE_COMP1(min,var,max) if(min>var || var>max){ CONSOLE("Error: "<<#var<<" out of range: "<<min<<" > "<<#var<<" = "<<var<<" > "<<max)
#define VERIFY_RANGE(min,var,max) __VERIFY_RANGE_COMP1(min,var,max)}
#define VERIFY_RANGE(min,var,max,ret)__VERIFY_RANGE_COMP1(min,var,max) ret;}

#define __VERIFY_BOOL_COMP1(val,comp,message) if(val != comp){CONSOLE("Error: "<<message)
#define VERIFY_BOOL(val,comp,message) __VERIFY_BOOL_COMP1(val,comp,message)}
#define VERIFY_BOOL(val,comp,message,ret) __VERIFY_BOOL_COMP1(val,comp,message) ret;}

#define __VERIFY_VALID_PTR_COMP1(ptr, message) if(!ptr){CONSOLE("Error: "<<#ptr<<" == nullltr "<<message)
#define VERIFY_VALID_PTR(ptr, message) __VERIFY_VALID_PTR_COMP1(ptr,message)}
#define VERIFY_VALID_PTR(ptr, message, ret) __VERIFY_VALID_PTR_COMP1(ptr,message) ret;}

