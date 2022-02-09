#pragma once
#include "CppUnitTest.h"
#include <cuda_runtime.h>
#include <Windows.h>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

extern void CUDA_validate(cudaError_t e);
extern void CUDA_validateLastError();


