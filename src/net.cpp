#include "..\inc\net.h"


// helper functions for cleaner time measuring code
std::chrono::time_point<std::chrono::high_resolution_clock> now() {
	return std::chrono::high_resolution_clock::now();
}

template <typename T>
double milliseconds(T t) {
	return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t).count() / 1000000;
}


Net::Net()
{
	m_inputSignalList	= nullptr;
	m_weightsList		= nullptr;
	m_outputSingalList  = nullptr;
	m_activationFunc	= nullptr;

	d_inputSignalList	= nullptr;
	d_weightsList		= nullptr;
	d_outputSingalList	= nullptr;

	m_inputs  = 0;
	m_hiddenX = 0;
	m_hiddenY = 0;
	m_outputs = 0;
	m_built   = false;
	m_neuronCount = 0;
	m_weightsCount = 0;
	setActivation(Activation::sigmoid);
	setHardware(Hardware::cpu);
}

Net::~Net()
{
	destroyDevice();
	m_built = false;
	if (m_inputSignalList) delete[] m_inputSignalList;
	if (m_outputSingalList) delete[] m_outputSingalList;

	m_inputSignalList = nullptr;
	m_outputSingalList = nullptr;
	destroyHostWeights();
}

void Net::setDimensions(size_t inputs, size_t hiddenX, size_t hiddenY, size_t outputs)
{
	if (m_built)
	{
		CONSOLE("Net is already built!")
		return;
	}
	m_inputs  = inputs;
	m_hiddenX = hiddenX;
	m_hiddenY = hiddenY;
	m_outputs = outputs;
}

size_t Net::getInputCount() const
{
	return m_inputs;
}
size_t Net::getHiddenXCount() const
{
	return m_hiddenX;
}
size_t Net::getHiddenYCount() const
{
	return m_hiddenY;
}
size_t Net::getOutputCount() const
{
	return m_outputs;
}

void Net::setActivation(Activation act)
{
	m_activation = act;
	switch (m_activation)
	{
		default:
			m_activation = Activation::sigmoid;
		case Activation::sigmoid:
			m_activationFunc = &Net::activation_sigmoid;
			return;
		case Activation::linear:
			m_activationFunc = &Net::activation_linear;
			return;
		case Activation::gauss:
			m_activationFunc = &Net::activation_gauss;
			return;
		
	}
}
Activation Net::getActivation() const
{
	return m_activation;
}

void Net::setHardware(Hardware ware)
{
	if (m_hardware == ware)
		return;
	//m_hardware = ware;
	CONSOLE("begin")
	auto t1 = now();
	switch (ware)
	{
		default:
			ware = Hardware::cpu;
		case Hardware::cpu:
		{
			if (!m_built)
				break;
			for (size_t i = 0; i < m_outputs; ++i)
				m_outputSingalList[i] = 0;
			CONSOLE("  Hardware: CPU")
			buildHostWeights();
			transferWeightsToHost();
			transferSignalsToHost();
			destroyDevice();
			

			break;
		}
		case Hardware::gpu_cuda:
		{
			if (!m_built)
				break;
			m_hardware = ware;
			for (size_t i = 0; i < m_outputs; ++i)
				m_outputSingalList[i] = 0;
			CONSOLE("  Hardware: GPU CUDA device")
			buildDevice();
			transferWeightsToDevice();
			//transferSignalsToDevice();
			destroyHostWeights();
		}
	}
	m_hardware = ware;
	auto t2 = now();
	CONSOLE("end. time: " << milliseconds(t2 - t1) << "ms")
}

Hardware Net::getHardware() const
{
	return m_hardware;
}

bool Net::build()
{
	CONSOLE("begin")
	auto t1 = now();
	bool success = true;

	success &= m_inputs > 0 && m_outputs > 0;

	if (success)
	{
		m_neuronCount = m_hiddenX * m_hiddenY + m_outputs;
		m_weightsCount = m_inputs * m_hiddenY + m_hiddenX * m_hiddenY * m_hiddenY + m_hiddenY * m_outputs;

		CONSOLE("  Inputs       : " << m_inputs)
		CONSOLE("  Hidden X     : " << m_hiddenX)
		CONSOLE("  Hidden Y     : " << m_hiddenY)
		CONSOLE("  Outputs      : " << m_outputs)

		CONSOLE("  Neuron  count: " << m_neuronCount)
		CONSOLE("  Weights count: " << m_weightsCount)
		CONSOLE("  Storage      : " << m_weightsCount * sizeof(float) + m_inputs * sizeof(float) + m_outputs * sizeof(float) << " bytes")
		m_inputSignalList = new float[m_inputs];
		m_outputSingalList = new float[m_outputs];
		for (size_t i = 0; i < m_outputs; ++i)
			m_outputSingalList[i] = 0;
		
		buildHostWeights();
		m_built = true;
		randomizeWeights();
		buildDevice();
		if (m_hardware != Hardware::cpu)
		{
			destroyHostWeights();
		}
		
	}

	auto t2 = now();
	CONSOLE("end. return "<<success<< " time: "<<milliseconds(t2-t1)<<"ms")
	return success;
}

void Net::randomizeWeights()
{
	if (!m_built) { CONSOLE("Error: build the net first") return; }
	randomizeWeights(0, m_weightsCount-1);
}
bool Net::randomizeWeights(size_t from, size_t to)
{
	VERIFY_BOOL(m_built, true, "build the net first", return false)
	if (from > to)
	{
		size_t tmp = to;
		to = from;
		from = tmp;
	}
	VERIFY_RANGE(0, to, m_weightsCount+1,return false)

	if ((to - from) > 1024)
	{
		NeuronalNet::GPU_CUDA_getRandomWeight(-1, 1, m_weightsList + from, to-from);
	}
	else
	{
		for (size_t i = from; i <= to; ++i)
		{
			m_weightsList[i] = getRandomValue(-1, 1);
		}
	}
	
	return true;
}
float Net::getRandomValue(float min, float max)
{
	//return 0.01;
	static bool first = true;
	if (first)
	{
		srand(time(NULL)); //seeding for the first time only!
		first = false;
	}
	float v1 = float(rand() % 100000) / 100000;

	if (min > max)
	{
		float tmp = max;
		max = min;
		min = tmp;
	}
	else if (size_t(min) == size_t(max))
		return min;

	return v1 + min + float(rand() % (size_t(max) - size_t(min)));
	
}

void Net::setInputVector(float* signalList)
{
	VERIFY_BOOL(m_built, true, "build the net first", return)
	for (size_t i = 0; i < m_inputs; ++i)
		m_inputSignalList[i] = signalList[i];
}

void Net::setInputVector(const SignalVector& signalList)
{
	VERIFY_BOOL(m_built, true, "build the net first", return)
	size_t max = signalList.size();
	if (m_inputs < max) max = m_inputs;
	for (size_t i = 0; i < max; ++i)
		m_inputSignalList[i] = signalList[i];
}

void Net::setInput(size_t index, float signal)
{
	VERIFY_BOOL(m_built,true,"build the net first",return)
	VERIFY_RANGE(0,index,m_inputs,return)
	m_inputSignalList[index] = signal;
}

float Net::getInput(size_t index) const
{
	VERIFY_BOOL(m_built, true, "build the net first", return 0)
	return m_inputSignalList[index];
}

std::vector<float> Net::getInputVector() const
{
	VERIFY_BOOL(m_built, true, "build the net first", return std::vector<float>())
	std::vector<float> inputVec(m_inputs);
	for (size_t i = 0; i < m_inputs; ++i)
		inputVec[i] = m_inputSignalList[i];
	return inputVec;
}

std::vector<float> Net::getOutputVector() const
{
	VERIFY_BOOL(m_built, true, "build the net first", return std::vector<float>())
		std::vector<float> outputVec(m_outputs);
	for (size_t i = 0; i < m_outputs; ++i)
		outputVec[i] = m_outputSingalList[i];
	return outputVec;
}

void Net::calculate()
{
	CONSOLE("begin")
	auto t1 = now();
	switch (m_hardware)
	{
		case Hardware::cpu:
		{
			CPU_calculate();
			break;
		}
		case Hardware::gpu_cuda:
		{
			GPU_CUDA_calculate();
			break;
		}
		default:
		{
			CONSOLE("Error: hardware undefined")
		}
	}
	auto t2 = now();
	CONSOLE("end. time: " << milliseconds(t2 - t1) << "ms")
}


void Net::CPU_calculate()
{
	CPU_calculateNet(m_weightsList, m_inputSignalList, m_outputSingalList,
					 m_inputs, m_hiddenX, m_hiddenY, m_outputs, m_activationFunc);
}

void Net::GPU_CUDA_calculate()
{
	transferSignalsToDevice();
	NeuronalNet::GPU_CUDA_calculateNet(d_weightsList, d_inputSignalList, d_outputSingalList,
									   m_inputs, m_hiddenX, m_hiddenY, m_outputs,m_activation);
	transferSignalsToHost();
}

void Net::CPU_calculateNet(float* weights, float* signals, float* outpuSignals,
					   size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, ActFp* activation)
{
	VERIFY_VALID_PTR(weights, "", return)
	VERIFY_VALID_PTR(signals, "", return)
	VERIFY_VALID_PTR(outpuSignals, "", return)
	VERIFY_VALID_PTR(activation, "", return)


	float* tmpHiddenOutSignals = new float[hiddenY];
	CPU_calculateLayer(weights, signals, tmpHiddenOutSignals, hiddenY, inputCount, activation);
	weights += inputCount * hiddenY;


	for (size_t i = 1; i < hiddenX; ++i)
	{
		CPU_calculateLayer(weights, tmpHiddenOutSignals, tmpHiddenOutSignals, hiddenY, hiddenY, activation);
		weights += hiddenY * hiddenY;
	}


	CPU_calculateLayer(weights, tmpHiddenOutSignals, outpuSignals, outputCount, hiddenY, activation);
	delete[] tmpHiddenOutSignals;
}
void Net::CPU_calculateLayer(float* weights, float* inputSignals, float* outputSignals,
							 size_t neuronCount, size_t inputSignalCount, ActFp* activation)
{
	float* tmpSignals = new float[inputSignalCount];
	memcpy(tmpSignals, inputSignals, inputSignalCount * sizeof(float));
	for (size_t index = 0; index < neuronCount; ++index)
	{
		float res = 0;
		for (size_t i = 0; i < inputSignalCount; ++i)
		{
			res += weights[index * inputSignalCount + i] * tmpSignals[i];
		}
		outputSignals[index] = (*activation)(res);
	}
	delete[] tmpSignals;
}
void Net::transferWeightsToDevice()
{
	switch (m_hardware)
	{
		case Hardware::gpu_cuda:
		{
			NeuronalNet::GPU_CUDA_transferToDevice(d_weightsList, m_weightsList, m_weightsCount * sizeof(float));
			break;
		}
		default:
			CONSOLE("Nothing to transfer")
	}
}
void Net::transferWeightsToHost()
{
	switch (m_hardware)
	{
		case Hardware::gpu_cuda:
		{
			NeuronalNet::GPU_CUDA_transferToHost(d_weightsList, m_weightsList, m_weightsCount * sizeof(float));
			break;
		}
		default:
			CONSOLE("Nothing to transfer")
	}
}
void Net::transferSignalsToDevice()
{
	switch (m_hardware)
	{
		case Hardware::gpu_cuda:
		{
			NeuronalNet::GPU_CUDA_transferToDevice(d_inputSignalList, m_inputSignalList, m_inputs * sizeof(float));
			break;
		}
		default:
			CONSOLE("Nothing to transfer")
	}
}
void Net::transferSignalsToHost()
{
	switch (m_hardware)
	{
		case Hardware::gpu_cuda:
		{
			NeuronalNet::GPU_CUDA_transferToHost(d_outputSingalList, m_outputSingalList, m_outputs * sizeof(float));
			break;
		}
		default:
			CONSOLE("Nothing to transfer")
	}
}

void Net::buildDevice()
{
	switch (m_hardware)
	{
		case Hardware::gpu_cuda:
		{
			NeuronalNet::GPU_CUDA_allocMem(d_inputSignalList, m_inputs * sizeof(float));
			NeuronalNet::GPU_CUDA_allocMem(d_weightsList, m_weightsCount * sizeof(float));
			NeuronalNet::GPU_CUDA_allocMem(d_outputSingalList, m_outputs * sizeof(float));
			break;
		}
		default: {}
	}
}
void Net::destroyDevice()
{
	switch (m_hardware)
	{
		case Hardware::gpu_cuda:
		{
			NeuronalNet::GPU_CUDA_freeMem(d_inputSignalList);
			NeuronalNet::GPU_CUDA_freeMem(d_weightsList);
			NeuronalNet::GPU_CUDA_freeMem(d_outputSingalList);
			break;
		}
		default: {}
	}
	
}
void Net::buildHostWeights()
{	
	m_weightsList = new float[m_weightsCount];
}
void Net::destroyHostWeights()
{
	if (m_weightsList) delete[] m_weightsList;
	m_weightsList = nullptr;
}
inline float Net::activation_linear(float inp)
{
	return NET_ACTIVATION_LINEAR(inp);
}
inline float Net::activation_gauss(float inp)
{
	//https://www.wolframalpha.com/input/?i=exp%28-pow%28x%2C2%29%29*2-1
	return NET_ACTIVATION_GAUSSIAN(inp);
}
inline float Net::activation_sigmoid(float inp) 
{
	return NET_ACTIVATION_SIGMOID(inp);
}