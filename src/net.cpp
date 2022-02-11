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
	NeuronalNet::GPU_CUDA_getSpecs();

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
	m_streamSize = 1;
	setActivation(Activation::sigmoid);
	setHardware(Hardware::cpu);
}

Net::~Net()
{
	destroyDevice();
	m_built = false;
	if (m_inputSignalList)
	{
		for (size_t i = 0; i < m_streamSize; ++i)
			delete[] m_inputSignalList[i];
		delete[] m_inputSignalList;
	}
	if (m_outputSingalList)
	{
		for (size_t i = 0; i < m_streamSize; ++i)
			delete[] m_outputSingalList[i];
		delete[] m_outputSingalList;
	}

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
void Net::setStreamSize(size_t size)
{
	m_streamSize = size;
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
			for (size_t i = 0; i < m_streamSize; ++i)
				memset(m_outputSingalList[i],0,m_outputs*sizeof(float));
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
			for (size_t i = 0; i < m_streamSize; ++i)
				memset(m_outputSingalList[i], 0, m_outputs * sizeof(float));
			CONSOLE("  Hardware: GPU CUDA device")
			buildDevice();
			transferWeightsToDevice();
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
	if (m_built)
		return 1;
	CONSOLE("begin")
	auto t1 = now();
	bool success = true;

	success &= m_inputs > 0 && m_outputs > 0;

	if (success)
	{
		m_neuronCount = m_hiddenX * m_hiddenY + m_outputs;
		if (m_hiddenX == 0 || m_hiddenY == 0)
		{
			m_hiddenX = 0; 
			m_hiddenY = 0;
			m_weightsCount = m_inputs * m_outputs;
		}
		else
			m_weightsCount = m_inputs * m_hiddenY + m_hiddenX * m_hiddenY * m_hiddenY + m_hiddenY * m_outputs;
		if (m_streamSize == 0)
			m_streamSize = 1;

		CONSOLE("  Inputs       : " << m_inputs)
		CONSOLE("  Hidden X     : " << m_hiddenX)
		CONSOLE("  Hidden Y     : " << m_hiddenY)
		CONSOLE("  Outputs      : " << m_outputs)


		CONSOLE("  Neuron  count: " << m_neuronCount)
		CONSOLE("  Weights count: " << m_weightsCount)
		CONSOLE("  Storage      : " << m_weightsCount * sizeof(float) + m_inputs * sizeof(float) + m_outputs * sizeof(float) << " bytes")
		CONSOLE("  Streams      : " << m_streamSize)

		m_inputSignalList = new float* [m_streamSize];
		m_outputSingalList = new float* [m_streamSize];
		for (size_t i = 0; i < m_streamSize; ++i)
		{
			m_inputSignalList[i] = new float[m_inputs];
			m_outputSingalList[i] = new float[m_outputs];

			memset(m_inputSignalList[i], 0, m_inputs * sizeof(float));
			memset(m_outputSingalList[i], 0, m_outputs * sizeof(float));
		}
		m_inputStream = MultiSignalVector(m_streamSize, SignalVector(m_inputs, 0));
		m_outputStream = MultiSignalVector(m_streamSize, SignalVector(m_outputs, 0));
		
		
		
		buildHostWeights();
		m_built = true;
		randomizeWeights();

		buildDevice();
		transferWeightsToDevice();
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
			//m_weightsList[i] = (float)i / 10;
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
	memcpy(m_inputSignalList[0], signalList, m_inputs * sizeof(float));
	m_inputStream[0] = std::vector<float>(signalList, signalList + m_inputs);
}
void Net::setInputVector(size_t stream, float* signalList)
{
	VERIFY_BOOL(m_built, true, "build the net first", return)
	VERIFY_RANGE(0, stream, m_streamSize - 1, return)
	memcpy(m_inputSignalList[stream], signalList, m_inputs * sizeof(float));
	m_inputStream[stream] = std::vector<float>(signalList, signalList + m_inputs);
}


void Net::setInputVector(const SignalVector& signalList)
{
	VERIFY_BOOL(m_built, true, "build the net first", return)
	size_t max = signalList.size();
	if (m_inputs < max) max = m_inputs;
	memcpy(m_inputSignalList[0], signalList.data(), max * sizeof(float));
}
void Net::setInputVector(size_t stream, const SignalVector& signalList)
{
	VERIFY_BOOL(m_built, true, "build the net first", return)
	VERIFY_RANGE(0, stream, m_streamSize - 1, return)
	size_t max = signalList.size();
	if (m_inputs < max) max = m_inputs;
	memcpy(m_inputSignalList[stream], signalList.data(), max * sizeof(float));
	m_inputStream[stream] = signalList;
}
void Net::setInputVector(const MultiSignalVector& streamVector)
{
	VERIFY_BOOL(m_built, true, "build the net first", return)
	VERIFY_RANGE(m_streamSize, streamVector.size(), m_streamSize, return)
	VERIFY_RANGE(m_inputs, streamVector[0].size(), m_inputs, return)
	size_t streams = streamVector.size();
	if (streams == 0)
		return;
	if (streams > m_streamSize)
		streams = m_streamSize;
	size_t max = streamVector[0].size();
	if (m_inputs < max) max = m_inputs;
	for (size_t s = 0; s < streams; ++s)
	{
		memcpy(m_inputSignalList[s], streamVector[s].data(), max * sizeof(float));
	}
	m_inputStream = streamVector;
}

void Net::setInput(size_t input, float signal)
{
	VERIFY_BOOL(m_built, true, "build the net first", return)
	VERIFY_RANGE(0, input, m_inputs-1, return)
	m_inputSignalList[0][input] = signal;
	m_inputStream[0][input] = signal;
}
void Net::setInput(size_t stream, size_t input, float signal)
{
	VERIFY_BOOL(m_built,true,"build the net first",return)
	VERIFY_RANGE(0, input,m_inputs-1,return)
	VERIFY_RANGE(0, stream, m_streamSize - 1, return)
	m_inputSignalList[stream][input] = signal;
	m_inputStream[stream][input] = signal;
}

float Net::getInput(size_t input) const
{
	VERIFY_BOOL(m_built, true, "build the net first", return 0)
	return m_inputStream[0][input];
	//return m_inputSignalList[0][input];
}
float Net::getInput(size_t stream, size_t input) const
{
	VERIFY_BOOL(m_built, true, "build the net first", return 0)
	VERIFY_RANGE(0, stream, m_streamSize-1, return 0)
	//return m_inputSignalList[stream][input];
	return m_inputStream[stream][input];
}

const SignalVector &Net::getInputVector(size_t stream)
{
	VERIFY_BOOL(m_built, true, "build the net first", return std::vector<float>())
	std::vector<float> inputVec(m_inputs);
	if (stream >= m_streamSize)
		stream = m_streamSize - 1;
	m_inputStream[stream] = std::vector<float>(m_inputSignalList[stream], m_inputSignalList[stream] + m_inputs);
	return m_inputStream[stream];
}
const MultiSignalVector& Net::getInputStreamVector()
{
	return m_inputStream;
}

const SignalVector &Net::getOutputVector(size_t stream) 
{
	VERIFY_BOOL(m_built, true, "build the net first", return std::vector<float>())
	
	if (stream >= m_streamSize)
		stream = m_streamSize - 1;
	m_outputStream[stream] = std::vector<float>(m_outputSingalList[stream], m_outputSingalList[stream] + m_outputs);
	return m_outputStream[stream];
}
const MultiSignalVector &Net::getOutputStreamVector() 
{
	for (size_t i = 0; i < m_streamSize; ++i)
	{
		m_outputStream[i] = std::vector<float>(m_outputSingalList[i], m_outputSingalList[i] + m_outputs);
	}
	return m_outputStream;
}

void Net::calculate()
{
	calculate(0, m_streamSize);
}
void Net::calculate(size_t stream)
{
	calculate(stream, stream+1);
}
void Net::calculate(size_t streamBegin, size_t streamEnd)
{
	if (streamBegin > streamEnd)
		std::swap(streamBegin, streamEnd);

	VERIFY_RANGE(0,streamBegin,m_streamSize,return)
	VERIFY_RANGE(0, streamEnd,m_streamSize,return)
	CONSOLE("begin")
		auto t1 = now();
	switch (m_hardware)
	{
		case Hardware::cpu:
		{
			CPU_calculate(streamBegin, streamEnd);
			break;
		}
		case Hardware::gpu_cuda:
		{
			GPU_CUDA_calculate(streamBegin, streamEnd);
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


void Net::CPU_calculate(size_t streamBegin, size_t streamEnd)
{
	for(size_t i= streamBegin; i< streamEnd; ++i)
		CPU_calculateNet(m_weightsList, m_inputSignalList[i], m_outputSingalList[i],
						 m_inputs, m_hiddenX, m_hiddenY, m_outputs, m_activationFunc);
}

void Net::GPU_CUDA_calculate(size_t streamBegin, size_t streamEnd)
{
	transferSignalsToDevice();
	NeuronalNet::GPU_CUDA_calculateNet(d_weightsList, d_inputSignalList+streamBegin, d_outputSingalList, streamEnd-streamBegin,
									   m_inputs, m_hiddenX, m_hiddenY, m_outputs,m_activation,
									   NeuronalNet::_d_cudaInfo);
	transferSignalsToHost();
}

void Net::CPU_calculateNet(float* weights, float* signals, float* outpuSignals,
					   size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, ActFp* activation)
{
	VERIFY_VALID_PTR(weights, "", return)
		VERIFY_VALID_PTR(signals, "", return)
		VERIFY_VALID_PTR(outpuSignals, "", return)
		VERIFY_VALID_PTR(activation, "", return)


		/*std::cout << "signals: ";
		for (size_t i = 0; i < inputCount; ++i)
			std::cout << signals[i] << "\t";
		std::cout << "\n";*/
	bool noHiddenLayer = !(hiddenY * hiddenX);

	if (noHiddenLayer)
	{
		CPU_calculateLayer(weights, signals, outpuSignals, outputCount, inputCount, activation);
	}
	else
	{
		float* tmpHiddenOutSignals1 = new float[hiddenY];
		float* tmpHiddenOutSignals2 = new float[hiddenY];

		CPU_calculateLayer(weights, signals, tmpHiddenOutSignals1, hiddenY, inputCount, activation);
		weights += inputCount * hiddenY;

		for (size_t i = 1; i < hiddenX; ++i)
		{
			CPU_calculateLayer(weights, tmpHiddenOutSignals1, tmpHiddenOutSignals2, hiddenY, hiddenY, activation);
			weights += hiddenY * hiddenY;
			float* tmp = tmpHiddenOutSignals1;
			tmpHiddenOutSignals1 = tmpHiddenOutSignals2;
			tmpHiddenOutSignals2 = tmp;
		}


		CPU_calculateLayer(weights, tmpHiddenOutSignals1, outpuSignals, outputCount, hiddenY, activation);
		delete[] tmpHiddenOutSignals1;
		delete[] tmpHiddenOutSignals2;
	}
	
}
void Net::CPU_calculateLayer(float* weights, float* inputSignals, float* outputSignals,
							 size_t neuronCount, size_t inputSignalCount, ActFp* activation)
{

	//std::cout << "Weights:\n";
	for (size_t index = 0; index < neuronCount; ++index)
	{
		float res = 0;
		
		for (size_t i = 0; i < inputSignalCount; ++i)
		{
			//printf("%3.2f ", weights[index * inputSignalCount + i]);
			res += weights[index * inputSignalCount + i] * inputSignals[i];
			//res += weights[index + inputSignalCount * i] * inputSignals[i];
		}
		//std::cout << "\n";
		outputSignals[index] = (*activation)(res);
	}
}
void Net::transferWeightsToDevice()
{
	switch (m_hardware)
	{
		case Hardware::gpu_cuda:
		{
			
			NeuronalNet::GPU_CUDA_transferToDevice(d_weightsList, m_weightsList, m_weightsCount * sizeof(float));
			NeuronalNet::GPU_CUDA_convertWeightMatrix(d_weightsList, m_inputs, m_hiddenX, m_hiddenY, m_outputs, NeuronalNet::Direction::toDevice);
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
			
			NeuronalNet::GPU_CUDA_convertWeightMatrix(d_weightsList, m_inputs, m_hiddenX, m_hiddenY, m_outputs,NeuronalNet::Direction::toHost);
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
			for (size_t i = 0; i < m_streamSize; ++i)
			{
				NeuronalNet::GPU_CUDA_transferToDevice(h_d_inputSignalList[i], m_inputSignalList[i], m_inputs * sizeof(float));
			}
			
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
			for (size_t i = 0; i < m_streamSize; ++i)
			{
				NeuronalNet::GPU_CUDA_transferToHost(h_d_outputStream[i], m_outputSingalList[i], m_outputs * sizeof(float));
			}
			//NeuronalNet::GPU_CUDA_transferToHost(d_outputSingalList, m_outputSingalList, m_outputs * sizeof(float));
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
			h_d_inputSignalList = new float* [m_streamSize];
			h_d_outputStream = new float* [m_streamSize];
			for (size_t i = 0; i < m_streamSize; ++i)
			{
				float* inpVec;
				float* outVec;
				NeuronalNet::GPU_CUDA_allocMem(inpVec, m_inputs * sizeof(float));
				NeuronalNet::GPU_CUDA_allocMem(outVec, m_outputs * sizeof(float));

				h_d_inputSignalList[i] = inpVec;
				h_d_outputStream[i] = outVec;
			}

			NeuronalNet::GPU_CUDA_allocMem(d_inputSignalList, m_streamSize * sizeof(float*));
			NeuronalNet::GPU_CUDA_allocMem(d_outputSingalList, m_outputs * sizeof(float*));
			NeuronalNet::GPU_CUDA_transferToDevice(d_inputSignalList, h_d_inputSignalList, m_streamSize * sizeof(float*));
			NeuronalNet::GPU_CUDA_transferToDevice(d_outputSingalList, h_d_outputStream, m_streamSize * sizeof(float*));

			NeuronalNet::GPU_CUDA_allocMem(d_weightsList, m_weightsCount * sizeof(float));
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
			NeuronalNet::GPU_CUDA_transferToHost(d_inputSignalList, h_d_inputSignalList, m_streamSize * sizeof(float*));
			NeuronalNet::GPU_CUDA_transferToHost(d_outputSingalList, h_d_outputStream, m_streamSize * sizeof(float*));

			for (size_t i = 0; i < m_streamSize; ++i)
			{
				NeuronalNet::GPU_CUDA_freeMem(h_d_inputSignalList[i]);
				NeuronalNet::GPU_CUDA_freeMem(h_d_outputStream[i]);
			}

			NeuronalNet::GPU_CUDA_freeMem(d_inputSignalList);
			NeuronalNet::GPU_CUDA_freeMem(d_weightsList);
			NeuronalNet::GPU_CUDA_freeMem(d_outputSingalList);
			delete[] h_d_inputSignalList;
			delete[] h_d_outputStream;
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