#include "backend/net.h"

namespace NeuronalNet
{
const SignalVector Net::m_emptySignalVectorDummy(0);
const MultiSignalVector Net::m_emptyMultiSignalVectorDummy(0,0);

size_t Net::m_net_instanceCount = 0;

Net::Net()
{
	++m_net_instanceCount;
	if(m_net_instanceCount == 1)
		NeuronalNet::GPU_CUDA_getSpecs();

	//m_inputSignalList	= nullptr;
	m_weightsList		= nullptr;
	//m_weightSignalProduct = nullptr;
	m_biasList			= nullptr;
	//m_outputSingalList  = nullptr;
	m_activationFunc	= nullptr;

	d_inputSignalList	= nullptr;
	h_d_inputSignalList = nullptr;
	d_netinputList		= nullptr;
	h_d_netinputList	= nullptr;
	d_neuronValueList	= nullptr;
	h_d_neuronValueList = nullptr;
	d_weightsList		= nullptr;
	d_weightSignalProduct = nullptr;
	h_d_weightSignalProduct = nullptr;
	d_biasList			= nullptr;
	d_outputSingalList	= nullptr;
	h_d_outputStream	= nullptr;

	m_inputs  = 0;
	m_hiddenX = 0;
	m_hiddenY = 0;
	m_outputs = 0;
	m_built   = false;
	m_useBias = true;
	m_neuronCount = 0;
	m_weightsCount = 0;
	m_streamSize = 1;

	m_weightsChangedFromDeviceTraining = false;
	m_biasChangedFromDeviceTraining = false;

	//m_useGraphics = false;
	setActivation(Activation::sigmoid);
	setHardware(Hardware::cpu);

}

Net::~Net()
{
	destroyDevice();
	m_built = false;
	/*if (m_inputSignalList)
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
	m_outputSingalList = nullptr;*/
	destroyHostWeights();
	destroyHostBias();

	if(m_net_instanceCount == 0)
		PRINT_ERROR("Impossible situation: net instance count can't be zero on this point because this incance does exist")
	else if (m_net_instanceCount == 1)
	{
		m_net_instanceCount = 0;
		NeuronalNet::GPU_CUDA_deleteSpecs();
	}
	else
	{
		--m_net_instanceCount;
	}
	
}

void Net::setDimensions(size_t inputs, size_t hiddenX, size_t hiddenY, size_t outputs)
{
	if (m_built)
	{
		PRINT_ERROR("Net is already built!")
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
size_t Net::getStreamSize() const
{
	return m_streamSize;
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
size_t Net::getNeuronCount() const
{
	return m_neuronCount;
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
			m_activationDerivetiveFunc = &Net::activation_sigmoid_derivetive;
			return;
		case Activation::linear:
			m_activationFunc = &Net::activation_linear;
			m_activationDerivetiveFunc = &Net::activation_linear_derivetive;
			return;
		case Activation::finiteLinear:
			m_activationFunc = &Net::activation_finiteLinear;
			m_activationDerivetiveFunc = &Net::activation_finiteLinear_derivetive;
			return;
		case Activation::binary:
			m_activationFunc = &Net::activation_binary;
			m_activationDerivetiveFunc = nullptr;
			return;
		case Activation::gauss:
			m_activationFunc = &Net::activation_gauss;
			m_activationDerivetiveFunc = &Net::activation_gauss_derivetive;
			return;
		
	}
	m_activationFunc = nullptr;
}
Activation Net::getActivation() const
{
	return m_activation;
}

void Net::setHardware(enum Hardware ware)
{
	DEBUG_FUNCTION_TIME_INTERVAL
	if (m_hardware == ware)
		return;
	
	//m_hardware = ware;
	//CONSOLE("begin")
	//auto t1 = now();
	switch (ware)
	{
		default:
			ware = Hardware::cpu;
		case Hardware::cpu:
		{
			if (!m_built)
				break;
			/*for (size_t i = 0; i < m_streamSize; ++i)
				memset(m_outputSingalList[i],0,m_outputs*sizeof(float));*/
			for(size_t i = 0; i < m_streamSize; ++i)
				memset(m_outputStream[i].begin(),0,m_outputs*sizeof(float));
			CONSOLE("Hardware: CPU")
			//buildHostWeights();
			//buildHostBias();
			m_netinputList = MultiSignalVector(m_streamSize, m_neuronCount);
			m_neuronValueList = MultiSignalVector(m_streamSize, m_neuronCount);
			transferWeightsToHost();
			transferBiasToHost();
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
				memset(m_outputStream[i].begin(), 0, m_outputs * sizeof(float));
			CONSOLE("Hardware: GPU CUDA device")
			buildDevice();
			transferWeightsToDevice();
			transferBiasToDevice();
			transferSignalsToDevice();
			m_netinputList.clear();
			m_neuronValueList.clear();
			//destroyHostWeights();
			//destroyHostBias();
			break;
		}
	}
	m_hardware = ware;
	//auto t2 = now();
	//CONSOLE("end. time: " << milliseconds(t2 - t1) << "ms")
}

Hardware Net::getHardware() const
{
	return m_hardware;
}
void Net::enableBias(bool enable)
{
	if (m_useBias == enable)
		return;
	m_useBias = enable;

	if (!m_built)
		return;
	if (!m_biasList)
		buildHostBias();
	if (m_useBias)
	{
		randomizeBias();
	}
	else
	{
		memset(m_biasList, 0, m_neuronCount * sizeof(float));
	}
	if (m_hardware != Hardware::cpu)
	{
		transferBiasToDevice();
		//destroyHostBias();
	}
}
bool Net::isBiasEnabled() const
{
	return m_useBias;
}

bool Net::build()
{
	DEBUG_FUNCTION_TIME_INTERVAL
	if (m_built)
		return 1;
	
	//CONSOLE("begin")
	//auto t1 = now();
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
			m_weightsCount = m_inputs * m_hiddenY + (m_hiddenX -1) * m_hiddenY * m_hiddenY + m_hiddenY * m_outputs;
		if (m_streamSize == 0)
			m_streamSize = 1;

		CONSOLE("Inputs       : " << m_inputs)
		CONSOLE("Hidden X     : " << m_hiddenX)
		CONSOLE("Hidden Y     : " << m_hiddenY)
		CONSOLE("Outputs      : " << m_outputs)


		CONSOLE("Neuron  count: " << m_neuronCount)
		CONSOLE("Weights count: " << m_weightsCount)
		CONSOLE("Storage      : " << Debug::bytesToString(m_weightsCount * sizeof(float) + m_inputs * sizeof(float) + m_outputs * sizeof(float)))
		CONSOLE("Streams      : " << m_streamSize)

		m_inputStream = MultiSignalVector(m_streamSize, m_inputs);
		m_outputStream = MultiSignalVector(m_streamSize, m_outputs);
		if (m_hardware == Hardware::cpu)
		{
			m_netinputList = MultiSignalVector(m_streamSize, m_neuronCount);
			m_neuronValueList = MultiSignalVector(m_streamSize, m_neuronCount);
		}
		
		
		
		buildHostWeights();
		buildHostBias();
		m_built = true;
		
		randomizeWeights();
		
		if (m_useBias)
		{
			randomizeBias();
		}
		else
		{
			memset(m_biasList, 0, m_neuronCount * sizeof(float));
		}
		
		buildDevice();
		transferWeightsToDevice();
		transferBiasToDevice();
		//if (m_hardware != Hardware::cpu)
		//{
		//	destroyHostWeights();
		//	destroyHostBias();
		//}
		
	}

	//auto t2 = now();
	//CONSOLE("end. return "<<success<< " time: "<<milliseconds(t2-t1)<<"ms")
	return success;
}
bool Net::isBuilt() const
{
	return m_built;
}
void Net::randomizeWeights()
{
	DEBUG_FUNCTION_TIME_INTERVAL
	if (!m_built) { PRINT_ERROR("Build the net first") return; }
	randomizeWeights(0, m_weightsCount);
}
bool Net::randomizeWeights(size_t from, size_t to)
{
	DEBUG_FUNCTION_TIME_INTERVAL
	VERIFY_BOOL(m_built, true, "build the net first", return false)
	if (from > to)
	{
		size_t tmp = to;
		to = from;
		from = tmp;
	}
	VERIFY_RANGE(0, to, m_weightsCount+1,return false)
/*
		for (size_t i = from; i <= to; ++i)
		{
			m_weightsList[i] = 1/(float)((i%10)+1)-0.5f;
		}
	return 1;*/
	randomize(m_weightsList + from, (to - from), -1, 1);
	if (m_hardware != Hardware::cpu && d_weightsList)
		transferWeightsToDevice();
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
void Net::randomizeBias()
{
	DEBUG_FUNCTION_TIME_INTERVAL
		if (!m_useBias)
			return;
	VERIFY_BOOL(m_built, true, "build the net first", return)
	randomize(m_biasList, m_neuronCount, -1, 1);
	if (m_hardware != Hardware::cpu && d_biasList)
		transferBiasToDevice();
}
void Net::randomize(float* list, size_t size, float min, float max)
{
	if (!list)
		return;

	if (size > 1024)
	{
		NeuronalNet::GPU_CUDA_getRandomValues(list, size, min, max);
	}
	else
	{
		for (size_t i = 0; i < size; ++i)
		{
			list[i] = getRandomValue(min, max);
		}
	}
}

void Net::setInputVector(float* signalList)
{
	DEBUG_FUNCTION_TIME_INTERVAL
	VERIFY_BOOL(m_built, true, "build the net first", return)
	//memcpy(m_inputSignalList[0], signalList, m_inputs * sizeof(float));
	m_inputStream[0].fill(signalList, m_inputs);// std::vector<float>(signalList, signalList + m_inputs);
}
void Net::setInputVector(size_t stream, float* signalList)
{
	DEBUG_FUNCTION_TIME_INTERVAL
	VERIFY_BOOL(m_built, true, "build the net first", return)
	VERIFY_RANGE(0, stream, m_streamSize - 1, return)
	//memcpy(m_inputSignalList[stream], signalList, m_inputs * sizeof(float));
	m_inputStream[stream].fill(signalList, m_inputs);// = std::vector<float>(signalList, signalList + m_inputs);
}


void Net::setInputVector(const SignalVector& signalList)
{
	DEBUG_FUNCTION_TIME_INTERVAL
	VERIFY_BOOL(m_built, true, "build the net first", return)
	size_t max = signalList.size();
	if (m_inputs < max) max = m_inputs;
	m_inputStream[0].fill(signalList.begin(), max);
	//memcpy(m_inputSignalList[0], signalList.begin(), max * sizeof(float));
}
void Net::setInputVector(size_t stream, const SignalVector& signalList)
{
	DEBUG_FUNCTION_TIME_INTERVAL
	VERIFY_BOOL(m_built, true, "build the net first", return)
	VERIFY_RANGE(0, stream, m_streamSize - 1, return)
	size_t max = signalList.size();
	if (m_inputs < max) max = m_inputs;
	//memcpy(m_inputSignalList[stream], signalList.begin(), max * sizeof(float));
	m_inputStream[stream].fill(signalList.begin(), max);
}
void Net::setInputVector(const MultiSignalVector& streamVector)
{
	DEBUG_FUNCTION_TIME_INTERVAL
	VERIFY_BOOL(m_built, true, "build the net first", return)
	VERIFY_RANGE(m_streamSize, streamVector.size(), m_streamSize, return)
	VERIFY_RANGE(m_inputs, streamVector.signalSize(), m_inputs, return)
	size_t streams = streamVector.size();
	if (streams == 0)
		return;
	if (streams > m_streamSize)
		streams = m_streamSize;
	size_t max = streamVector.signalSize();
	if (m_inputs < max) max = m_inputs;
	
	m_inputStream.fill(streamVector.begin(), streams);
	/*for (size_t s = 0; s < streams; ++s)
	{

		memcpy(m_inputSignalList[s], streamVector[s].begin(), max * sizeof(float));
	}*/
	//m_inputStream = streamVector;
}

void Net::setInput(size_t input, float signal)
{
	VERIFY_BOOL(m_built, true, "build the net first", return)
	VERIFY_RANGE(0, input, m_inputs-1, return)
	//m_inputSignalList[0][input] = signal;
	m_inputStream[0][input] = signal;
}
void Net::setInput(size_t stream, size_t input, float signal)
{
	VERIFY_BOOL(m_built,true,"build the net first",return)
	VERIFY_RANGE(0, input,m_inputs-1,return)
	VERIFY_RANGE(0, stream, m_streamSize - 1, return)
	//m_inputSignalList[stream][input] = signal;
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
	VERIFY_BOOL(m_built, true, "build the net first", return m_emptySignalVectorDummy)
	std::vector<float> inputVec(m_inputs);
	if (stream >= m_streamSize)
		stream = m_streamSize - 1;
	//m_inputStream[stream].fill(m_inputSignalList[stream], m_inputs);
	return m_inputStream[stream];
}
const MultiSignalVector& Net::getInputStreamVector()
{
	return m_inputStream;
}

const SignalVector &Net::getOutputVector(size_t stream) 
{
	VERIFY_BOOL(m_built, true, "build the net first", return m_emptySignalVectorDummy)
	
	if (stream >= m_streamSize)
		stream = m_streamSize - 1;
	//m_outputStream[stream].fill(m_outputSingalList[stream], m_outputs);
	return m_outputStream[stream];
}
const MultiSignalVector &Net::getOutputStreamVector() 
{
	/*for (size_t i = 0; i < m_streamSize; ++i)
	{
		m_outputStream[i].fill(m_outputSingalList[i], m_outputs);
	}*/
	return m_outputStream;
}
MultiSignalVector Net::getNetinputStreamVector() const
{
	switch (m_hardware)
	{
		case Hardware::cpu:
		{
			return m_netinputList;
		}
		case Hardware::gpu_cuda:
		{
			MultiSignalVector tmpVec(m_streamSize, m_neuronCount);
			for (size_t i = 0; i < m_streamSize; ++i)
			{
				//NeuronalNet::GPU_CUDA_transferToHost(h_d_outputStream[i], m_outputSingalList[i], m_outputs * sizeof(float));
				NeuronalNet::GPU_CUDA_transferToHost(h_d_netinputList[i], tmpVec[i].begin(), m_neuronCount * sizeof(float));
			}
			return tmpVec;
			break;
		}
		default:
		{
			PRINT_ERROR("Error: hardware undefined")
		}
	}
	return MultiSignalVector(0, 0);
}

MultiSignalVector Net::getNeuronValueStreamVector() const
{
	switch (m_hardware)
	{
		case Hardware::cpu:
		{
			return m_neuronValueList;
		}
		case Hardware::gpu_cuda:
		{
			MultiSignalVector tmpVec(m_streamSize, m_neuronCount);
			for (size_t i = 0; i < m_streamSize; ++i)
			{
				//NeuronalNet::GPU_CUDA_transferToHost(h_d_outputStream[i], m_outputSingalList[i], m_outputs * sizeof(float));
				NeuronalNet::GPU_CUDA_transferToHost(h_d_neuronValueList[i], tmpVec[i].begin(), m_neuronCount * sizeof(float));
			}
			return tmpVec;
			break;
		}
		default:
		{
			PRINT_ERROR("Error: hardware undefined")
		}
	}
	return MultiSignalVector(0, 0);
}

void Net::setWeight(size_t layer, size_t neuron, size_t input, float weight)
{
	VERIFY_RANGE(0, layer, m_hiddenX, return)
	size_t index;
	if(m_hiddenX * m_hiddenY == 0 || layer == 0)
	{
		index = m_inputs * neuron + input;
	}
	else
	{
		index = m_inputs * m_hiddenY + (layer - 1) * m_hiddenX * m_hiddenY + m_hiddenY * neuron + input;
	}
	VERIFY_RANGE(0, index, m_weightsCount, return)
	m_weightsList[index] = weight;
	switch (m_hardware)
	{
		case Hardware::cpu:
		{
			//m_weightsList[index] = weight;
			break;
		}
		case Hardware::gpu_cuda:
		{
			transferWeightsToDevice();
			//PRINT_ERROR("Not implemented for GPU yet, use CPU as device")
			break;
		}
		default:
		{PRINT_ERROR("Device not defined " << (int)m_hardware)}
	}
}
void Net::setWeight(const std::vector<float>& list)
{
	setWeight(list.data());
}
void Net::setWeight(const float* list)
{
	setWeight(list, 0, m_weightsCount);
}
void Net::setWeight(const float* list, size_t to)
{
	setWeight(list, 0, to);
}
void Net::setWeight(const float* list, size_t insertOffset, size_t count)
{
	memcpy(m_weightsList + insertOffset, list, count * sizeof(float));
	switch (m_hardware)
	{
		case Hardware::cpu:
		{
			//memcpy(m_weightsList + insertOffset, list, count * sizeof(float));
			break;
		}
		case Hardware::gpu_cuda:
		{
			transferWeightsToDevice();
			//PRINT_ERROR("Not implemented for GPU yet, use CPU as device")
			break;
		}
		default:
			PRINT_ERROR("Device not defined " << (int)m_hardware)
	}
}
float Net::getWeight(size_t layer, size_t neuron, size_t input)
{
	size_t index;
	if (m_hiddenX * m_hiddenY == 0 || layer == 0)
	{
		index = m_inputs * neuron + input;
	}
	else
	{
		index = m_inputs * m_hiddenY + (layer - 1) * m_hiddenX * m_hiddenY + m_hiddenY * neuron + input;
	}
	VERIFY_RANGE(0, index, m_weightsCount, return 0)
	if (m_weightsChangedFromDeviceTraining)
		transferWeightsToHost();
	return m_weightsList[index];
	/*switch (m_hardware)
	{
		case Hardware::cpu:
		{
			return m_weightsList[index];
			break;
		}
		case Hardware::gpu_cuda:
		{
			PRINT_ERROR("Not implemented for GPU yet, use CPU as device")
				break;
		}
		default:
			PRINT_ERROR("Device not defined " << (int)m_hardware)
	}
	return 0;*/
}
const float* Net::getWeight()
{
	if (m_weightsChangedFromDeviceTraining)
		transferWeightsToHost();
	return m_weightsList;
	/*switch (m_hardware)
	{
		case Hardware::cpu:
		{
			return m_weightsList;
		}
		case Hardware::gpu_cuda:
		{
			PRINT_ERROR("Memory is on device, therefore not available for host code")
			return nullptr;
		}
		default:
			PRINT_ERROR("Device not defined " << (int)m_hardware)
	}
	return nullptr;*/
}
size_t Net::getWeightSize() const
{
	return m_weightsCount;
}
const float* Net::getBias()
{
	if (m_biasChangedFromDeviceTraining)
		transferBiasToHost();
	return m_biasList;
}


void Net::calculate()
{
	DEBUG_FUNCTION_TIME_INTERVAL
	calculate(0, m_streamSize);
}
void Net::calculate(size_t stream)
{
	DEBUG_FUNCTION_TIME_INTERVAL
	calculate(stream, stream+1);
}
void Net::calculate(size_t streamBegin, size_t streamEnd)
{
	DEBUG_FUNCTION_TIME_INTERVAL
	if (streamBegin > streamEnd)
		std::swap(streamBegin, streamEnd);

	VERIFY_RANGE(0,streamBegin,m_streamSize,return)
	VERIFY_RANGE(0, streamEnd,m_streamSize,return)
	//CONSOLE("begin")
	//	auto t1 = now();
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
			PRINT_ERROR("Error: hardware undefined")
		}
	}
	//auto t2 = now();
	//CONSOLE("end. time: " << milliseconds(t2 - t1) << "ms")
}

/*
void Net::addGraphics(GraphicsNeuronInterface* obj)
{
	if (obj == nullptr)
		return;
	for (size_t i = 0; i < m_graphicsNeuronInterfaceList.size(); ++i)
		if (m_graphicsNeuronInterfaceList[i] == obj)
			return;
	m_useGraphics = true;
	m_graphicsNeuronInterfaceList.push_back(obj);
}
void Net::removeGraphics(GraphicsNeuronInterface* obj)
{
	for (size_t i = 0; i < m_graphicsNeuronInterfaceList.size(); ++i)
		if (m_graphicsNeuronInterfaceList[i] == obj)
		{
			m_graphicsNeuronInterfaceList.erase(m_graphicsNeuronInterfaceList.begin() + i);
			return;
		}
}
void Net::addGraphics(GraphicsConnectionInterface* obj)
{
	if (obj == nullptr)
		return;
	for (size_t i = 0; i < m_graphicsConnectionInterfaceList.size(); ++i)
		if (m_graphicsConnectionInterfaceList[i] == obj)
			return;
	m_useGraphics = true;
	m_graphicsConnectionInterfaceList.push_back(obj);
}
void Net::removeGraphics(GraphicsConnectionInterface* obj)
{
	for (size_t i = 0; i < m_graphicsConnectionInterfaceList.size(); ++i)
		if (m_graphicsConnectionInterfaceList[i] == obj)
		{
			m_graphicsConnectionInterfaceList.erase(m_graphicsConnectionInterfaceList.begin() + i);
			return;
		}
}
void Net::clearGraphics()
{
	m_graphicsNeuronInterfaceList.clear();
	m_graphicsConnectionInterfaceList.clear();
	m_useGraphics = false;
}*/
void Net::graphics_update(const vector<GraphicsNeuronInterface*>& graphicsNeuronInterfaceList,
						  const vector<GraphicsConnectionInterface*>& graphicsConnectionInterfaceList,
						  size_t streamIndex)
{
	//static vector<double> timeBench(1000, 0);
	//static size_t indx = 0;
	//Debug::Timer dbgTimer(true);
	if (streamIndex >= m_streamSize)
	{
		PRINT_ERROR("streamIndex is out of range")
			return;
	}

	switch (m_hardware)
	{
		case Hardware::cpu:
		{
			graphics_update_CPU(graphicsNeuronInterfaceList, streamIndex);
			graphics_update_CPU(graphicsConnectionInterfaceList, streamIndex);

			break;
		}
		case Hardware::gpu_cuda:
		{
			graphics_update_GPU_CUDA(graphicsNeuronInterfaceList, streamIndex);
			graphics_update_GPU_CUDA(graphicsConnectionInterfaceList, streamIndex);
			break;
		}
		default:
		{
			PRINT_ERROR("Error: hardware undefined")
		}
	}
}
void Net::graphics_update_CPU(const vector<GraphicsNeuronInterface*>& objList, size_t streamIndex)
{
	float minN = m_neuronValueList[streamIndex][getMinIndex(m_neuronValueList[streamIndex].begin(), m_neuronCount)];
	float maxN = m_neuronValueList[streamIndex][getMaxIndex(m_neuronValueList[streamIndex].begin(), m_neuronCount)];

	float minO = -1.f;
	float maxO = 1.f;

	for (GraphicsNeuronInterface* obj : objList)
	{
		graphics_update(obj, minN, maxN, minO, maxO,
						m_inputStream[streamIndex].begin(),
						m_neuronValueList[streamIndex].begin(),
						m_netinputList[streamIndex].begin());
	}
	/*NeuronIndex index = obj->index();
	size_t neuronIndex;
	float neuronOutput;
	float netinput;
	
	switch (index.type)
	{
		case NeuronType::input:
		{
#ifdef NET_GRAPHICS_ERRORCHECK
			if (index.y >= m_inputs)
			{
				graphics_outOfRange(obj);
				return;
			}
#endif
			neuronOutput = m_inputStream[streamIndex][index.y];
			netinput = neuronOutput;
			break;
		}
		case NeuronType::hidden:
		{
#ifdef NET_GRAPHICS_ERRORCHECK
			if (index.y >= m_hiddenY || 
				index.x >= m_hiddenX)
			{
				graphics_outOfRange(obj);
				return;
			}
#endif
			neuronIndex = index.x * m_hiddenY + index.y;
			neuronOutput = m_neuronValueList[streamIndex][neuronIndex];
			netinput = m_netinputList[streamIndex][neuronIndex];
			break;
		}
		case NeuronType::output:
		{
#ifdef NET_GRAPHICS_ERRORCHECK
			if (index.y >= m_outputs)
			{
				graphics_outOfRange(obj);
				return;
			}
#endif
			neuronIndex = m_hiddenX * m_hiddenY + index.y;
			neuronOutput = m_neuronValueList[streamIndex][neuronIndex];
			netinput = m_netinputList[streamIndex][neuronIndex];
			break;
		}
		default:
		{
#ifdef NET_GRAPHICS_ERRORCHECK
			graphics_outOfRange(obj);
#endif
			return;
		}
	}

	obj->update(netinput, neuronOutput,minN,maxN,minO,maxO);*/
}
void Net::graphics_update_GPU_CUDA(const vector<GraphicsNeuronInterface*> &objList, size_t streamIndex)
{
	
	

	float* netinputData = new float[m_neuronCount];
	float* neuronOutputs = new float[m_neuronCount];

	//NeuronalNet::GPU_CUDA_transferToHost(h_d_outputStream[streamIndex], m_outputStream[i].begin(), m_outputs * sizeof(float));

	NeuronalNet::GPU_CUDA_transferToHost(h_d_netinputList[streamIndex], netinputData, m_neuronCount * sizeof(float));
	NeuronalNet::GPU_CUDA_transferToHost(h_d_neuronValueList[streamIndex], neuronOutputs, m_neuronCount * sizeof(float));

	float minO = -1.f;
	float maxO = 1.f;
	float minN = netinputData[getMinIndex(netinputData, m_neuronCount)];
	float maxN = netinputData[getMaxIndex(netinputData, m_neuronCount)];

	for (GraphicsNeuronInterface* obj : objList)
	{
		graphics_update(obj, minN, maxN, minO, maxO,
						m_inputStream[streamIndex].begin(), neuronOutputs, netinputData);
	}
	delete[] neuronOutputs;
	delete[] netinputData;

}
void Net::graphics_update(GraphicsNeuronInterface* obj,
						  float minN, float maxN, float minO, float maxO,
						  float* inputSignals,  float* neuronOutputData, float* netinputData)
{
	NeuronIndex index = obj->index();
	size_t neuronIndex;
	float neuronOutput;
	float netinput;

	switch (index.type)
	{
		case NeuronType::input:
		{
#ifdef NET_GRAPHICS_ERRORCHECK
			if (index.y >= m_inputs)
			{
				graphics_outOfRange(obj);
				return;
			}
#endif
			neuronOutput = inputSignals[index.y];
			netinput = neuronOutput;
			break;
		}
		case NeuronType::hidden:
		{
#ifdef NET_GRAPHICS_ERRORCHECK
			if (index.y >= m_hiddenY ||
				index.x >= m_hiddenX)
			{
				graphics_outOfRange(obj);
				return;
			}
#endif
			neuronIndex = index.x * m_hiddenY + index.y;
			neuronOutput = neuronOutputData[neuronIndex];
			netinput = netinputData[neuronIndex];
			break;
		}
		case NeuronType::output:
		{
#ifdef NET_GRAPHICS_ERRORCHECK
			if (index.y >= m_outputs)
			{
				graphics_outOfRange(obj);
				return;
			}
#endif
			neuronIndex = m_hiddenX * m_hiddenY + index.y;
			neuronOutput = neuronOutputData[neuronIndex];
			netinput = netinputData[neuronIndex];
			break;
		}
		default:
		{
#ifdef NET_GRAPHICS_ERRORCHECK
			graphics_outOfRange(obj);
#endif
			return;
		}
	}

	obj->update(netinput, neuronOutput, minN, maxN, minO, maxO);
}
void Net::graphics_outOfRange(GraphicsNeuronInterface* obj)
{
	NeuronIndex index = obj->index();
	PRINT_ERROR("NeuronIndex is out of range : type: " << typeToString(index.type) <<
			" x = " << index.x << " y = " << index.y);
}
void Net::graphics_update_CPU(const vector<GraphicsConnectionInterface*>& objList, size_t streamIndex)
{
	float minW = m_weightsList[getMinIndex(m_weightsList, m_weightsCount)];
	float maxW = m_weightsList[getMaxIndex(m_weightsList, m_weightsCount)];

	float minS = -1.f;
	float maxS = 1.f;
	for (GraphicsConnectionInterface* obj : objList)
	{
		graphics_update(obj, minW, maxW, minS, maxS,
						m_weightsList, m_weightSignalProduct[streamIndex].begin());
	}
	/*ConnectionIndex index = obj->index();
	size_t weightIndex = 0;
	//size_t signalNeuronIndex;
	
	float neuronOutput = 0;
	float weight = 0;
	float signal = 0;

	switch (index.neuron.type)
	{
		case NeuronType::input:
		{
			// Connections not possible on a input neuron
#ifdef NET_GRAPHICS_ERRORCHECK
				graphics_outOfRange(obj);
#endif
				return;
			break;
		}
		case NeuronType::hidden:
		{
#ifdef NET_GRAPHICS_ERRORCHECK
			if (index.neuron.y >= m_hiddenY ||
				index.neuron.x >= m_hiddenX)
			{
				graphics_outOfRange(obj);
				return;
			}
#endif
			if (index.neuron.x == 0)
			{
#ifdef NET_GRAPHICS_ERRORCHECK
				if (index.inputConnection >= m_inputs)
				{
					graphics_outOfRange(obj);
					return;
				}
#endif
				weightIndex = m_inputs * index.neuron.y + 
							  index.inputConnection;
				neuronOutput = m_inputStream[streamIndex][index.inputConnection];

			}

			else
			{
#ifdef NET_GRAPHICS_ERRORCHECK
				if (index.inputConnection >= m_hiddenY)
				{
					graphics_outOfRange(obj);
					return;
				}
#endif
				
				weightIndex = m_inputs * m_hiddenY + 
							  (index.neuron.x -1) * m_hiddenY * m_hiddenY +
					          m_hiddenY * index.neuron.y + index.inputConnection;
				neuronOutput = m_neuronValueList[streamIndex][(index.neuron.x-1) * m_hiddenY + index.inputConnection];
			}

			weight = m_weightsList[weightIndex];
			signal = weight * neuronOutput;

			//neuronIndex = index.neuron.x * m_hiddenY + index.neuron.y;
			//neuronOutput = m_neuronValueList[stramIndex][neuronIndex];
			//netinput = m_netinputList[stramIndex][neuronIndex];
			break;
		}
		case NeuronType::output:
		{
			
#ifdef NET_GRAPHICS_ERRORCHECK
			if (index.neuron.y >= m_outputs)
			{
				graphics_outOfRange(obj);
				return;
			}
#endif
			if (m_hiddenX == 0)
			{
#ifdef NET_GRAPHICS_ERRORCHECK
				if (index.inputConnection >= m_inputs)
				{
					graphics_outOfRange(obj);
					return;
				}
#endif
				weightIndex = m_inputs * index.neuron.y +
					          index.inputConnection;
				neuronOutput = m_inputStream[streamIndex][index.inputConnection];

				weight = m_weightsList[weightIndex];
				signal = weight * neuronOutput;
			}
			else
			{
#ifdef NET_GRAPHICS_ERRORCHECK
				if (index.inputConnection >= m_hiddenY)
				{
					graphics_outOfRange(obj);
					return;
				}
#endif
				weightIndex = m_inputs * m_hiddenY +
					(m_hiddenX - 1) * m_hiddenY * m_hiddenY +
					m_hiddenY * index.neuron.y + index.inputConnection;
				neuronOutput = m_neuronValueList[streamIndex][(m_hiddenX -1) * m_hiddenY + index.inputConnection];
				weight = m_weightsList[weightIndex];
				signal = weight * neuronOutput;
			}
			break;
		}
		default:
		{
#ifdef NET_GRAPHICS_ERRORCHECK
			graphics_outOfRange(obj);
#endif
			return;
		}
	}

	

	obj->update(weight, signal,minW,maxW,minS,minS);*/
}
void Net::graphics_update_GPU_CUDA(const vector<GraphicsConnectionInterface*> &objList, size_t streamIndex)
{
#define graphics_update_GPU_CUDA_LAZY

#ifdef graphics_update_GPU_CUDA_LAZY
	//float* weights = new float[m_weightsCount];
	float* weightsSignals = new float[m_weightsCount];

	//NeuronalNet::GPU_CUDA_transferToHost(h_d_outputStream[streamIndex], m_outputStream[i].begin(), m_outputs * sizeof(float));

	//float* d_tmpW;
	//NeuronalNet::GPU_CUDA_allocMem(d_tmpW, m_weightsCount * sizeof(float));
	//NeuronalNet::GPU_CUDA_memcpy(d_weightsList, d_tmpW, m_weightsCount * sizeof(float));
	//NeuronalNet::GPU_CUDA_convertWeightMatrix(d_tmpW, m_inputs, m_hiddenX, m_hiddenY, m_outputs, NeuronalNet::Direction::toHost);
	//NeuronalNet::GPU_CUDA_transferToHost(d_tmpW, weights, m_weightsCount * sizeof(float));

	NeuronalNet::GPU_CUDA_transferToHost(h_d_weightSignalProduct[streamIndex], weightsSignals, m_weightsCount * sizeof(float));
	
	float minW = getWeight()[getMinIndex(m_weightsList, m_weightsCount)];
	float maxW = getWeight()[getMaxIndex(m_weightsList, m_weightsCount)];

	float minS = -1.f;
	float maxS = 1.f;

	for (GraphicsConnectionInterface* obj : objList)
	{
		graphics_update(obj, minW, maxW, minS, maxS,
						m_weightsList, weightsSignals);
	}

	//NeuronalNet::GPU_CUDA_freeMem(d_tmpW);
	//delete[] weights;
	delete[] weightsSignals;
#else

#endif
}
inline void Net::graphics_update(GraphicsConnectionInterface* obj,
							float minW, float maxW, float minS, float maxS,
							float* weightList, float* connectionSignalList)
{
	ConnectionIndex index = obj->index();
	size_t weightIndex = 0;
	//size_t signalNeuronIndex;

	float weight = 0;
	float signal = 0;

	switch (index.neuron.type)
	{
		case NeuronType::input:
		{
			// Connections not possible on a input neuron
#ifdef NET_GRAPHICS_ERRORCHECK
			graphics_outOfRange(obj);
#endif
			return;
			break;
		}
		case NeuronType::hidden:
		{
#ifdef NET_GRAPHICS_ERRORCHECK
			if (index.neuron.y >= m_hiddenY ||
				index.neuron.x >= m_hiddenX)
			{
				graphics_outOfRange(obj);
				return;
			}
#endif
			if (index.neuron.x == 0)
			{
#ifdef NET_GRAPHICS_ERRORCHECK
				if (index.inputConnection >= m_inputs)
				{
					graphics_outOfRange(obj);
					return;
				}
#endif
				weightIndex = m_inputs * index.neuron.y +
					index.inputConnection;
				//neuronOutput = m_inputStream[streamIndex][index.inputConnection];

			}

			else
			{
#ifdef NET_GRAPHICS_ERRORCHECK
				if (index.inputConnection >= m_hiddenY)
				{
					graphics_outOfRange(obj);
					return;
				}
#endif

				weightIndex = m_inputs * m_hiddenY +
					(index.neuron.x - 1) * m_hiddenY * m_hiddenY +
					m_hiddenY * index.neuron.y + index.inputConnection;
				//neuronOutput = m_neuronValueList[streamIndex][(index.neuron.x - 1) * m_hiddenY + index.inputConnection];
			}

			weight = weightList[weightIndex];
			signal = connectionSignalList[weightIndex];

			//neuronIndex = index.neuron.x * m_hiddenY + index.neuron.y;
			//neuronOutput = m_neuronValueList[stramIndex][neuronIndex];
			//netinput = m_netinputList[stramIndex][neuronIndex];
			break;
		}
		case NeuronType::output:
		{

#ifdef NET_GRAPHICS_ERRORCHECK
			if (index.neuron.y >= m_outputs)
			{
				graphics_outOfRange(obj);
				return;
			}
#endif
			if (m_hiddenX == 0)
			{
#ifdef NET_GRAPHICS_ERRORCHECK
				if (index.inputConnection >= m_inputs)
				{
					graphics_outOfRange(obj);
					return;
				}
#endif
				weightIndex = m_inputs * index.neuron.y +
					index.inputConnection;
				//neuronOutput = m_inputStream[streamIndex][index.inputConnection];

				weight = weightList[weightIndex];
				signal = connectionSignalList[weightIndex];
			}
			else
			{
#ifdef NET_GRAPHICS_ERRORCHECK
				if (index.inputConnection >= m_hiddenY)
				{
					graphics_outOfRange(obj);
					return;
				}
#endif
				weightIndex = m_inputs * m_hiddenY +
					(m_hiddenX - 1) * m_hiddenY * m_hiddenY +
					m_hiddenY * index.neuron.y + index.inputConnection;
				//neuronOutput = m_neuronValueList[streamIndex][(m_hiddenX - 1) * m_hiddenY + index.inputConnection];
				weight = weightList[weightIndex];
				signal = connectionSignalList[weightIndex];
			}
			break;
		}
		default:
		{
#ifdef NET_GRAPHICS_ERRORCHECK
			graphics_outOfRange(obj);
#endif
			return;
		}
	}



	obj->update(weight, signal, minW, maxW, minS, minS);
}
void Net::graphics_outOfRange(GraphicsConnectionInterface* obj)
{
	ConnectionIndex index = obj->index();
	PRINT_ERROR("Connection is out of range: type: " << typeToString(index.neuron.type) <<
			" x = " << index.neuron.x << " y = " << index.neuron.y << " Input: "<< index.inputConnection);
}
void Net::CPU_calculate(size_t streamBegin, size_t streamEnd)
{
	DEBUG_FUNCTION_TIME_INTERVAL
		for (size_t i = streamBegin; i < streamEnd; ++i)
			//CPU_calculateNet(m_weightsList, m_inputSignalList[i], m_outputSingalList[i],
			//				 m_inputs, m_hiddenX, m_hiddenY, m_outputs, m_activationFunc);
			CPU_calculateNet(m_weightsList, m_biasList, m_inputStream[i].begin(), m_outputStream[i].begin(),
							 m_weightSignalProduct[i].begin(), m_netinputList[i].begin(), m_neuronValueList[i].begin(),
							 m_inputs, m_hiddenX, m_hiddenY, m_outputs, m_activationFunc);
} 

void Net::GPU_CUDA_calculate(size_t streamBegin, size_t streamEnd)
{
	DEBUG_FUNCTION_TIME_INTERVAL
	transferSignalsToDevice();
	NeuronalNet::GPU_CUDA_calculateNet(d_weightsList, d_biasList, d_inputSignalList+streamBegin, d_outputSingalList + streamBegin,
									   d_weightSignalProduct + streamBegin, d_netinputList + streamBegin, d_neuronValueList + streamBegin, streamEnd-streamBegin,
									   m_inputs, m_hiddenX, m_hiddenY, m_outputs,m_activation,
									   NeuronalNet::_d_cudaInfo);
	transferSignalsToHost();
}

void Net::CPU_calculateNet(float* weights, float* biasList, float* signals, float* outpuSignals, 
						   float* connectionSignals, float* netinputList, float* neuronSignalList,
					       size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, ActFp* activation)
{
	DEBUG_FUNCTION_TIME_INTERVAL
	PTR_CHECK_NULLPTR(weights, return)
	PTR_CHECK_NULLPTR(biasList, return)
	PTR_CHECK_NULLPTR(signals, return)
	PTR_CHECK_NULLPTR(outpuSignals, return)
	PTR_CHECK_NULLPTR(activation,return)


	/*std::cout << "signals: ";
	for (size_t i = 0; i < inputCount; ++i)
		std::cout << signals[i] << "\t";
	std::cout << "\n";*/
	bool noHiddenLayer = !(hiddenY * hiddenX);

	if (noHiddenLayer)
	{
		CPU_calculateLayer(weights, biasList, signals,
						   connectionSignals, netinputList, neuronSignalList,
						   outputCount, inputCount, activation);
	}
	else
	{
		float* tmpHiddenOutSignals1 = neuronSignalList;
		//float* tmpHiddenOutSignals2 = neuronSignalList;

		CPU_calculateLayer(weights, biasList, signals, 
						   connectionSignals, netinputList, neuronSignalList,
						   hiddenY, inputCount, activation);
		size_t wDeltaIndex = inputCount * hiddenY;
		weights += wDeltaIndex;
		connectionSignals += wDeltaIndex;
		netinputList += hiddenY;
		neuronSignalList += hiddenY;
		biasList += hiddenY;

		for (size_t i = 1; i < hiddenX; ++i)
		{
			//tmpHiddenOutSignals2 = neuronSignalList;
			CPU_calculateLayer(weights, biasList, tmpHiddenOutSignals1, 
							   connectionSignals, netinputList, neuronSignalList,
							   hiddenY, hiddenY, activation);
			wDeltaIndex = hiddenY * hiddenY;
			weights += wDeltaIndex;
			connectionSignals += wDeltaIndex;
			netinputList += hiddenY;
			tmpHiddenOutSignals1 = neuronSignalList;
			neuronSignalList += hiddenY;
			biasList += hiddenY;
			//float* tmp = tmpHiddenOutSignals1;
			//tmpHiddenOutSignals1 = tmpHiddenOutSignals2;
			//tmpHiddenOutSignals2 = tmp;
		}


		CPU_calculateLayer(weights, biasList, tmpHiddenOutSignals1, 
						   connectionSignals, netinputList, neuronSignalList,
						   outputCount, hiddenY, activation);
		//delete[] tmpHiddenOutSignals1;
		//delete[] tmpHiddenOutSignals2;
	}
	memcpy(outpuSignals, neuronSignalList, outputCount * sizeof(float));
	
}
void Net::CPU_calculateLayer(float* weights, float* biasList, float* inputSignals, 
							 float* connectionSignals, float* netinputList, float* neuronSignalList,
							 size_t neuronCount, size_t inputSignalCount, ActFp* activation)
{
	//DEBUG_FUNCTION_TIME_INTERVAL
	//std::cout << "Weights:\n";
	for (size_t index = 0; index < neuronCount; ++index)
	{
		float res = 0;
		//double res_ = 0;
		for (size_t i = 0; i < inputSignalCount; ++i)
		//for (size_t i = 0; i < 1; ++i)
		{
			//printf("%3.2f ", weights[index * inputSignalCount + i]);
			//res += inputSignals[i];


			//res += weights[index * inputSignalCount + i] * inputSignals[i];
			size_t wIndex = index * inputSignalCount + i;
			connectionSignals[wIndex] = weights[wIndex] * inputSignals[i];
			res += connectionSignals[wIndex];



			//res_ += (double)weights[index * inputSignalCount + i] * (double)inputSignals[i];
			
			
			
			/*float delta = (float)round((double)(weights[index * inputSignalCount + i] * inputSignals[i] * 1000.l)) / 1000.f;
			uint32_t* valueBits = (uint32_t*)&delta;

			*valueBits = *valueBits & 0xffff0000;
			res += *(float*)valueBits;*/
		}
		//res = (float)res_;
		res += biasList[index];
		res = round(res * 100.f) / 100.f;
		netinputList[index] = res;
		//if (index == 0)
		//	printf("C: %f\n", res);
		//outputSignals[index] = round(res * 100.f) / 100.f;
		neuronSignalList[index] = round((*activation)(res) * 100.f) / 100.f;

	}
}
void Net::transferWeightsToDevice()
{
	DEBUG_FUNCTION_TIME_INTERVAL
	switch (m_hardware)
	{
		case Hardware::gpu_cuda:
		{
			//NeuronalNet::GPU_CUDA_transferToDevice(d_biasList, m_biasList, m_neuronCount * sizeof(float));
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
	DEBUG_FUNCTION_TIME_INTERVAL
	switch (m_hardware)
	{
		case Hardware::gpu_cuda:
		{
			
			float* tmpW = nullptr;
			NeuronalNet::GPU_CUDA_allocMem(tmpW, m_weightsCount * sizeof(float));
			NeuronalNet::GPU_CUDA_memcpy(d_weightsList, tmpW, m_weightsCount * sizeof(float));
			NeuronalNet::GPU_CUDA_convertWeightMatrix(tmpW, m_inputs, m_hiddenX, m_hiddenY, m_outputs,NeuronalNet::Direction::toHost);
			NeuronalNet::GPU_CUDA_transferToHost(d_weightsList, m_weightsList, m_weightsCount * sizeof(float));
			NeuronalNet::GPU_CUDA_freeMem(tmpW);
			break;
		}
		default:
			CONSOLE("Nothing to transfer")
	}
	m_weightsChangedFromDeviceTraining = false;
}
void Net::transferSignalsToDevice()
{
	DEBUG_FUNCTION_TIME_INTERVAL
	switch (m_hardware)
	{
		case Hardware::gpu_cuda:
		{
			for (size_t i = 0; i < m_streamSize; ++i)
			{
				//NeuronalNet::GPU_CUDA_transferToDevice(h_d_inputSignalList[i], m_inputSignalList[i], m_inputs * sizeof(float));
				NeuronalNet::GPU_CUDA_transferToDevice(h_d_inputSignalList[i], m_inputStream[i].begin(), m_inputs * sizeof(float));
				if (m_netinputList.size() > 0)
					NeuronalNet::GPU_CUDA_transferToDevice(h_d_netinputList[i], m_netinputList[i].begin(), m_neuronCount * sizeof(float));
				if(m_neuronValueList.size() >0)
					NeuronalNet::GPU_CUDA_transferToDevice(h_d_neuronValueList[i], m_neuronValueList[i].begin(), m_neuronCount * sizeof(float));
				if (m_weightSignalProduct.size() > 0)
				{
					NeuronalNet::GPU_CUDA_transferToDevice(h_d_weightSignalProduct[i], m_weightSignalProduct[i].begin(), m_weightsCount * sizeof(float));
					NeuronalNet::GPU_CUDA_convertWeightMatrix(h_d_weightSignalProduct[i], m_inputs, m_hiddenX, m_hiddenY, m_outputs, NeuronalNet::Direction::toDevice);
				}
			}
			break;
		}
		default:
			CONSOLE("Nothing to transfer")
	}
}
void Net::transferSignalsToHost()
{
	DEBUG_FUNCTION_TIME_INTERVAL
	switch (m_hardware)
	{
		case Hardware::gpu_cuda:
		{
			for (size_t i = 0; i < m_streamSize; ++i)
			{
				//NeuronalNet::GPU_CUDA_transferToHost(h_d_outputStream[i], m_outputSingalList[i], m_outputs * sizeof(float));
				NeuronalNet::GPU_CUDA_transferToHost(h_d_outputStream[i], m_outputStream[i].begin(), m_outputs * sizeof(float));
				if (m_netinputList.size() > 0)
					NeuronalNet::GPU_CUDA_transferToHost(h_d_netinputList[i], m_netinputList[i].begin(), m_neuronCount * sizeof(float));
				if(m_neuronValueList.size() > 0)
					NeuronalNet::GPU_CUDA_transferToHost(h_d_neuronValueList[i], m_neuronValueList[i].begin(), m_neuronCount * sizeof(float));
				if (m_weightSignalProduct.size() > 0)
				{
					NeuronalNet::GPU_CUDA_convertWeightMatrix(h_d_weightSignalProduct[i], m_inputs, m_hiddenX, m_hiddenY, m_outputs, NeuronalNet::Direction::toHost);
					NeuronalNet::GPU_CUDA_transferToHost(h_d_weightSignalProduct[i], m_weightSignalProduct[i].begin(), m_weightsCount * sizeof(float));
				}
			}
			
			//NeuronalNet::GPU_CUDA_transferToHost(d_outputSingalList, m_outputSingalList, m_outputs * sizeof(float));
			break;
		}
		default:
			CONSOLE("Nothing to transfer")
	}
}
void Net::transferBiasToDevice()
{
	DEBUG_FUNCTION_TIME_INTERVAL
	switch (m_hardware)
	{
		case Hardware::gpu_cuda:
		{
			NeuronalNet::GPU_CUDA_transferToDevice(d_biasList, m_biasList, m_neuronCount * sizeof(float));
			break;
		}
		default:
			CONSOLE("Nothing to transfer")
	}
}
void Net::transferBiasToHost()
{
	DEBUG_FUNCTION_TIME_INTERVAL
	switch (m_hardware)
	{
		case Hardware::gpu_cuda:
		{
			NeuronalNet::GPU_CUDA_transferToHost(d_biasList, m_biasList, m_neuronCount * sizeof(float));
			break;
		}
		default:
			CONSOLE("Nothing to transfer")
	}
	m_biasChangedFromDeviceTraining = false;
}

void Net::buildDevice()
{
	DEBUG_FUNCTION_TIME_INTERVAL
	switch (m_hardware)
	{
		case Hardware::gpu_cuda:
		{
			h_d_inputSignalList = DBG_NEW float* [m_streamSize];
			h_d_outputStream = DBG_NEW float* [m_streamSize];
			h_d_netinputList = DBG_NEW float* [m_streamSize];
			h_d_neuronValueList = DBG_NEW float* [m_streamSize];
			h_d_weightSignalProduct = DBG_NEW float* [m_streamSize];
			for (size_t i = 0; i < m_streamSize; ++i)
			{
				float* inpVec = nullptr;
				float* outVec = nullptr;
				float* netinputVec = nullptr;
				float* neuronValueVec = nullptr;
				float* connectionSigVec = nullptr;
				NeuronalNet::GPU_CUDA_allocMem(inpVec, m_inputs * sizeof(float));
				NeuronalNet::GPU_CUDA_allocMem(outVec, m_outputs * sizeof(float));
				NeuronalNet::GPU_CUDA_allocMem(netinputVec, m_neuronCount * sizeof(float));
				NeuronalNet::GPU_CUDA_allocMem(neuronValueVec, m_neuronCount * sizeof(float));
				NeuronalNet::GPU_CUDA_allocMem(connectionSigVec, m_weightsCount * sizeof(float));

				h_d_inputSignalList[i] = inpVec;
				h_d_outputStream[i] = outVec;
				h_d_netinputList[i] = netinputVec;
				h_d_neuronValueList[i] = neuronValueVec;
				h_d_weightSignalProduct[i] = connectionSigVec;
			}

			NeuronalNet::GPU_CUDA_allocMem(d_inputSignalList, m_streamSize * sizeof(float*));
			NeuronalNet::GPU_CUDA_allocMem(d_outputSingalList, m_streamSize * sizeof(float*));
			NeuronalNet::GPU_CUDA_allocMem(d_netinputList, m_streamSize * sizeof(float*));
			NeuronalNet::GPU_CUDA_allocMem(d_neuronValueList, m_streamSize * sizeof(float*));
			NeuronalNet::GPU_CUDA_allocMem(d_weightSignalProduct, m_streamSize * sizeof(float*));

			NeuronalNet::GPU_CUDA_transferToDevice(d_inputSignalList, h_d_inputSignalList, m_streamSize * sizeof(float*));
			NeuronalNet::GPU_CUDA_transferToDevice(d_outputSingalList, h_d_outputStream, m_streamSize * sizeof(float*));
			NeuronalNet::GPU_CUDA_transferToDevice(d_netinputList, h_d_netinputList, m_streamSize * sizeof(float*));
			NeuronalNet::GPU_CUDA_transferToDevice(d_neuronValueList, h_d_neuronValueList, m_streamSize * sizeof(float*));
			NeuronalNet::GPU_CUDA_transferToDevice(d_weightSignalProduct, h_d_weightSignalProduct, m_streamSize * sizeof(float*));

			NeuronalNet::GPU_CUDA_allocMem(d_weightsList, m_weightsCount * sizeof(float));
			NeuronalNet::GPU_CUDA_allocMem(d_biasList, m_neuronCount * sizeof(float));

			
			break;
		}
		default: {}
	}
}
void Net::destroyDevice()
{
	DEBUG_FUNCTION_TIME_INTERVAL
	switch (m_hardware)
	{
		case Hardware::gpu_cuda:
		{
			NeuronalNet::GPU_CUDA_transferToHost(d_inputSignalList, h_d_inputSignalList, m_streamSize * sizeof(float*));
			NeuronalNet::GPU_CUDA_transferToHost(d_outputSingalList, h_d_outputStream, m_streamSize * sizeof(float*));
			NeuronalNet::GPU_CUDA_transferToHost(d_netinputList, h_d_netinputList, m_streamSize * sizeof(float*));
			NeuronalNet::GPU_CUDA_transferToHost(d_neuronValueList, h_d_neuronValueList, m_streamSize * sizeof(float*));
			NeuronalNet::GPU_CUDA_transferToHost(d_weightSignalProduct, h_d_weightSignalProduct, m_streamSize * sizeof(float*));

			for (size_t i = 0; i < m_streamSize; ++i)
			{
				NeuronalNet::GPU_CUDA_freeMem(h_d_inputSignalList[i]);
				NeuronalNet::GPU_CUDA_freeMem(h_d_outputStream[i]);
				NeuronalNet::GPU_CUDA_freeMem(h_d_netinputList[i]);
				NeuronalNet::GPU_CUDA_freeMem(h_d_neuronValueList[i]);
				NeuronalNet::GPU_CUDA_freeMem(h_d_weightSignalProduct[i]);
			}

			NeuronalNet::GPU_CUDA_freeMem(d_inputSignalList);
			NeuronalNet::GPU_CUDA_freeMem(d_weightsList);
			NeuronalNet::GPU_CUDA_freeMem(d_biasList);
			NeuronalNet::GPU_CUDA_freeMem(d_outputSingalList);
			NeuronalNet::GPU_CUDA_freeMem(d_netinputList);
			NeuronalNet::GPU_CUDA_freeMem(d_neuronValueList);
			NeuronalNet::GPU_CUDA_freeMem(d_weightSignalProduct);
			delete[] h_d_inputSignalList;
			delete[] h_d_outputStream;
			delete[] h_d_netinputList;
			delete[] h_d_neuronValueList;
			delete[] h_d_weightSignalProduct;

			h_d_inputSignalList = nullptr;
			h_d_outputStream = nullptr;
			h_d_netinputList = nullptr;
			h_d_neuronValueList = nullptr;
			h_d_weightSignalProduct = nullptr;
			break;
		}
		default: {}
	}
	
}
void Net::buildHostWeights()
{	
	DEBUG_FUNCTION_TIME_INTERVAL
	m_weightsList	= DBG_NEW float[m_weightsCount];
	m_weightSignalProduct.resize(m_streamSize, m_weightsCount);
}
void Net::buildHostBias()
{
	m_biasList = DBG_NEW float[m_neuronCount];
}
void Net::destroyHostWeights()
{
	DEBUG_FUNCTION_TIME_INTERVAL
	if (m_weightsList) delete[] m_weightsList;
	m_weightsList = nullptr;
	m_weightSignalProduct.clear();
}
void Net::destroyHostBias()
{
	if (m_biasList) delete[] m_biasList;
	m_biasList = nullptr;
}
float Net::activation_linear(float inp)
{
	return NET_ACTIVATION_LINEAR(inp);
}
float Net::activation_finiteLinear(float inp)
{
	return NET_ACTIVATION_FINITELINEAR(inp);
}
float Net::activation_binary(float inp)
{
	return NET_ACTIVATION_BINARY(inp);
}
float Net::activation_gauss(float inp)
{
	//https://www.wolframalpha.com/input/?i=exp%28-pow%28x%2C2%29%29*2-1
	return NET_ACTIVATION_GAUSSIAN(inp);
}
float Net::activation_sigmoid(float inp) 
{
	return NET_ACTIVATION_SIGMOID(inp);
}

float Net::activation_linear_derivetive(float inp)
{
	return NET_ACTIVATION_LINEAR_DERIVETIVE(inp);
}
float Net::activation_finiteLinear_derivetive(float inp)
{
	return NET_ACTIVATION_FINITELINEAR_DERIVETIVE(inp);
}
float Net::activation_gauss_derivetive(float inp)
{
	//https://www.wolframalpha.com/input/?i=exp%28-pow%28x%2C2%29%29*2-1
	return NET_ACTIVATION_GAUSSIAN_DERIVETIVE(inp);
}
float Net::activation_sigmoid_derivetive(float inp)
{
	return NET_ACTIVATION_SIGMOID_DERIVETIVE(inp);
}
};