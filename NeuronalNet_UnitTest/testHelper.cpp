#include "testHelper.h"
#include "neuronalNet.h"


//using namespace NeuronalNet;
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
	for (size_t i = 0; i < NeuronalNet::Debug::_unitTest_consoleBuffer.size(); ++i)
	{
		Logger::WriteMessage(NeuronalNet::Debug::_unitTest_consoleBuffer[i].c_str());
	}
	Logger::WriteMessage("------------CONSOLE PLOT END------------\n");
	NeuronalNet::Debug::_unitTest_consoleBuffer.clear();
}

void printSignal(const NeuronalNet::SignalVector& sig)
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
void printSignal(const NeuronalNet::MultiSignalVector& sig)
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


bool signalEqual(const NeuronalNet::SignalVector& a, const NeuronalNet::SignalVector& b)
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
bool signalEqual(const NeuronalNet::MultiSignalVector& a, const NeuronalNet::MultiSignalVector& b)
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

void getNetData(NeuronalNet::Net &net,
				std::vector<float>& weights,
				std::vector<float>& bias)
{
	const float* w = net.getWeight();
	weights.clear();
	weights.reserve(net.getWeightSize());
	for (size_t i = 0; i < net.getWeightSize(); ++i)
		weights.push_back(w[i]);

	const float* b = net.getBias();
	bias.clear();
	bias.reserve(net.getNeuronCount());
	for (size_t i = 0; i < net.getNeuronCount(); ++i)
		bias.push_back(b[i]);
}