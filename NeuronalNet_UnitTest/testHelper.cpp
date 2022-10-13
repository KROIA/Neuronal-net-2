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

void plotConsoleOutput()
{
	Logger::WriteMessage("--------------CONSOLE PLOT--------------\n");
	for (size_t i = 0; i < Debug::_unitTest_consoleBuffer.size(); ++i)
	{
		Logger::WriteMessage(Debug::_unitTest_consoleBuffer[i].c_str());
	}
	Logger::WriteMessage("------------CONSOLE PLOT END------------\n");
	Debug::_unitTest_consoleBuffer.clear();
}

void printSignal(const SignalVector& sig)
{
	Logger::WriteMessage("Signals: ");
	for (size_t i = 0; i < sig.size(); ++i)
	{
		char str[20];
		sprintf_s(str,"%f\t", sig[i]);
		Logger::WriteMessage(str);
	}
	Logger::WriteMessage("\n");
}
void printSignal(const MultiSignalVector& sig)
{
	Logger::WriteMessage("Streams:\n");
	for (size_t i = 0; i < sig.size(); ++i)
	{
		char str[20];
		sprintf_s(str, " Stream [%5li]\t", i);
		Logger::WriteMessage(str);
		printSignal(sig[i]);
	}
	Logger::WriteMessage("\n");
}


bool signalEqual(const SignalVector& a, const SignalVector& b)
{
	if (a.size() != b.size())
		return false;

	for (size_t i = 0; i < a.size(); ++i)
	{
		if (abs(a[i] - b[i]) > 0.00001)
			return false;
	}
	return true;
}
bool signalEqual(const MultiSignalVector& a, const MultiSignalVector& b)
{
	if (a.size() != b.size())
		return false;

	for (size_t i = 0; i < a.size(); ++i)
	{
		if (!signalEqual(a[i], b[i]))
			return false;
	}
	return true;
}