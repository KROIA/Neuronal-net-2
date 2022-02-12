#pragma once
#include "CppUnitTest.h"
#include <cuda_runtime.h>
#include <Windows.h>
#include <string>
#include "pch.h"


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


extern void printSignal(const class SignalVector& sig);
extern void printSignal(const class MultiSignalVector& sig);
extern bool signalEqual(const class SignalVector& a, const class SignalVector& b);
extern bool signalEqual(const class MultiSignalVector& a, const class MultiSignalVector& b);