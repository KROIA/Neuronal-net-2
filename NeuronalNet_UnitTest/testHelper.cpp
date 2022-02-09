#include "testHelper.h"

void CUDA_validate(cudaError_t e)
{
	if (e != cudaError_t::cudaSuccess)
	{
		std::wstring w;
		const char* message = cudaGetErrorString(e);
		std::copy(message, message + strlen(message), back_inserter(w));
		const WCHAR* pwcsName = w.c_str();
		Assert::Fail(pwcsName);
	}
}

void CUDA_validateLastError()
{
	CUDA_validate(cudaGetLastError());
}


