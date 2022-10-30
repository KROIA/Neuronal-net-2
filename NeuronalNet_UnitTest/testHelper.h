#pragma once
#include "CppUnitTest.h"
#include <cuda_runtime.h>
#include <Windows.h>
#include <string>
//#include "pch.h"
#include "neuronalNet.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

extern void CUDA_validate(cudaError_t e);
extern void CUDA_validateLastError();

extern void plotConsoleOutput();
class PostTestConsolePlotter
{
	public:
	PostTestConsolePlotter() {};
	~PostTestConsolePlotter() { plotConsoleOutput(); }
};
#define POST_CONSOLE_PLOT PostTestConsolePlotter __plotter;


extern void printSignal(const NeuronalNet::SignalVector& sig);
extern void printSignal(const NeuronalNet::MultiSignalVector& sig);
extern bool signalEqual(const NeuronalNet::SignalVector& a, const NeuronalNet::SignalVector& b);
extern bool signalEqual(const NeuronalNet::MultiSignalVector& a, const NeuronalNet::MultiSignalVector& b);

extern void getNetData(NeuronalNet::Net &net,
					   std::vector<float>& weights,
					   std::vector<float>& bias);

