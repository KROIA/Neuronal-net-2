

#include "backend/backpropNet.h"
#ifdef USE_CUDA
#include "net_kernel.cuh"
#endif
namespace NeuronalNet
{
	BackpropNet::BackpropNet()
	{
		m_learnParameter		= 1;
		h_d_outputDifference	= nullptr;
		d_outputDifference		= nullptr;
		d_deltaWeight			= nullptr;
		h_d_deltaWeight			= nullptr;
		d_deltaBias				= nullptr;
		h_d_deltaBias			= nullptr;
		d_expected				= nullptr;
		h_d_expected			= nullptr;
	}
	BackpropNet::~BackpropNet()
	{
		//deltaWeight.clear();
		//deltaBias.clear();
	}

	/*void BackpropNet::setHardware(enum Hardware ware)
	{
		Net::setHardware(ware);
		switch (m_hardware)
		{
			case Hardware::cpu:
			{
				if (!m_built)
					break;
				for (size_t i = 0; i < m_streamSize; ++i)
					memset(m_outputStream[i].begin(), 0, m_outputs * sizeof(float));
				CONSOLE("Hardware: CPU")
					buildHostWeights();
				buildHostBias();
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
				destroyHostWeights();
				destroyHostBias();
			}
			default:
			{

			}
		}
	}*/
	bool BackpropNet::build()
	{
		bool ret = Net::build();
		m_expected.resize(m_streamSize, m_outputs);
		m_expectedChanged.resize(m_streamSize, true);
		m_outputDifference.resize(m_streamSize,m_outputs);
        m_deltaWeight.resize(m_streamSize, m_weightsCount);
        m_deltaBias.resize(m_streamSize, m_neuronCount);
		return ret;
	}

	void BackpropNet::setLearnParameter(float learnParam)
	{
		m_learnParameter = learnParam;
	}
	float BackpropNet::getLearnParameter() const
	{
		return m_learnParameter;
	}

	void BackpropNet::setExpectedOutput(const MultiSignalVector& expectedOutputVec)
	{
		DEBUG_BENCHMARK_STACK
		DEBUG_FUNCTION_TIME_INTERVAL
		VERIFY_RANGE(0, expectedOutputVec.size(), m_streamSize, return)
		VERIFY_BOOL(expectedOutputVec[0].size(), m_outputs, "expectedOutputVec[0].size() != m_outputs", return)
		m_expected = expectedOutputVec;
		m_expectedChanged = vector<bool>(m_expected.size(), true);
	}
	void BackpropNet::setExpectedOutput(const SignalVector& expectedOutputVec)
	{
		DEBUG_BENCHMARK_STACK
		DEBUG_FUNCTION_TIME_INTERVAL
		VERIFY_BOOL(expectedOutputVec.size(), m_outputs, "expectedOutputVec.size() != m_outputs", return)
		m_expected[0] = expectedOutputVec;
		m_expectedChanged[0] = true;
	}
	void BackpropNet::setExpectedOutput(size_t streamIndex, const SignalVector& expectedOutputVec)
	{
		DEBUG_BENCHMARK_STACK
		DEBUG_FUNCTION_TIME_INTERVAL
		VERIFY_RANGE(0, streamIndex, m_streamSize, return)
		VERIFY_BOOL(expectedOutputVec.size(), m_outputs, "expectedOutputVec.size() != m_outputs", return)
		m_expected[streamIndex] = expectedOutputVec;
		m_expectedChanged[streamIndex] = true;
	}
	void BackpropNet::learn()
	{
		switch (m_hardware)
		{
			case Hardware::cpu:
			{
				CPU_learn();
				break;
			}
#ifdef USE_CUDA
			case Hardware::gpu_cuda:
			{
				GPU_learn();
				break;
			}
#endif
		}
	}
	void BackpropNet::learn(size_t streamIndex)
	{
		switch (m_hardware)
		{
			case Hardware::cpu:
			{
				//float* deltaW = new float[m_weightsCount];
				//float* deltaB = new float[m_neuronCount];
				//
				//memset(deltaW, 0, m_weightsCount * sizeof(float));
				//memset(deltaB, 0, m_neuronCount * sizeof(float));
				memset(m_deltaWeight[streamIndex].begin(), 0, m_weightsCount * sizeof(float));
				memset(m_deltaBias[streamIndex].begin(), 0, m_neuronCount * sizeof(float));
				CPU_learn(streamIndex, m_deltaWeight[streamIndex].begin(), m_deltaBias[streamIndex].begin());

				for (size_t w = 0; w < m_weightsCount; ++w)
				{
					m_weightsList[w] += m_deltaWeight[streamIndex][w] * m_learnParameter;
				}
				for (size_t b = 0; b < m_neuronCount; ++b)
				{
					m_biasList[b] += m_deltaBias[streamIndex][b] * m_learnParameter;
				}
				//delete[] deltaW;
				//delete[] deltaB;
				break;
			}
#ifdef USE_CUDA
			case Hardware::gpu_cuda:
			{
				GPU_learn(streamIndex);
				break;
			}
#endif
		}
	}
	void BackpropNet::learn(const MultiSignalVector& expectedOutputVec)
	{
		DEBUG_BENCHMARK_STACK
		DEBUG_FUNCTION_TIME_INTERVAL
		VERIFY_RANGE(0, expectedOutputVec.size(), m_streamSize, return)
		VERIFY_BOOL(expectedOutputVec[0].size(), m_outputs, "expectedOutputVec[0].size() != m_outputs", return)
		setExpectedOutput(expectedOutputVec);
		learn();
	}
	void BackpropNet::learn(const SignalVector& expectedOutputVec)
	{
		DEBUG_BENCHMARK_STACK
		DEBUG_FUNCTION_TIME_INTERVAL
		VERIFY_BOOL(expectedOutputVec.size(), m_outputs, "expectedOutputVec.size() != m_outputs", return)
		setExpectedOutput(expectedOutputVec);
		learn(0);
	}
	void BackpropNet::learn(size_t streamIndex,  const SignalVector& expectedOutputVec)
	{
		DEBUG_BENCHMARK_STACK
		DEBUG_FUNCTION_TIME_INTERVAL
		VERIFY_RANGE(0, streamIndex, m_streamSize, return)
		VERIFY_BOOL(expectedOutputVec.size(), m_outputs, "expectedOutputVec.size() != m_outputs", return)
		setExpectedOutput(streamIndex, expectedOutputVec);
		learn(streamIndex);
	}
	const SignalVector& BackpropNet::getError(size_t streamIndex)
	{
		VERIFY_RANGE(0, streamIndex, m_streamSize-1,return m_outputDifference[0])

		return internal_getError(streamIndex);
	}
	const MultiSignalVector& BackpropNet::getError(const MultiSignalVector& expectedOutputVec)
	{
		size_t maxStreamSize = m_streamSize;
		if (expectedOutputVec.size() != m_streamSize)
		{
			CONSOLE_FUNCTION("Warning: expectedOutputVec.size() != m_streamSize")

			if (expectedOutputVec.size() < m_streamSize)
			{
				maxStreamSize = expectedOutputVec.size();
			}
			
			for (size_t i = 0; i < maxStreamSize; ++i)
				getError(i, expectedOutputVec[i]);
			return m_outputDifference;
		}

		m_expected = expectedOutputVec;
		m_expectedChanged = vector<bool>(m_expected.size(), true);
		return internal_getError();
	}
	const SignalVector& BackpropNet::getError(size_t streamIndex, const SignalVector& expectedOutputVec)
	{
		DEBUG_BENCHMARK_STACK
		VERIFY_RANGE(0, streamIndex, m_streamSize - 1, return m_outputDifference[0])
		m_expected[streamIndex] = expectedOutputVec;
		m_expectedChanged[streamIndex] = true;
		return internal_getError(streamIndex);
	}
	const MultiSignalVector& BackpropNet::getError() const
	{
		switch (m_hardware)
		{
#ifdef USE_CUDA
			case Hardware::gpu_cuda:
			{
				
				//GPU_CUDA_learnBackpropagation_getOutputError(h_d_outputStream[streamIndex], expectedOutputVec.begin(), m_outputDifference[streamIndex].begin(), m_outputs);
				for (size_t i = 0; i < m_streamSize; ++i)
					GPU_CUDA_transferToHost(h_d_outputDifference[i], m_outputDifference[i].begin(), m_outputs * sizeof(float));
				break;
			}
#endif
			default:
			{

			}
			case Hardware::cpu: {}
		}
		return m_outputDifference;
	}

	void BackpropNet::buildDevice()
	{
		DEBUG_FUNCTION_TIME_INTERVAL
		Net::buildDevice();
		

		switch (m_hardware)
		{
#ifdef USE_CUDA
			case Hardware::gpu_cuda:
			{
				h_d_outputDifference	= DBG_NEW float* [m_streamSize];
				h_d_deltaWeight			= DBG_NEW float* [m_streamSize];
				h_d_deltaBias			= DBG_NEW float* [m_streamSize];
				h_d_expected			= DBG_NEW float* [m_streamSize];

				for (size_t i = 0; i < m_streamSize; ++i)
				{
					float* out;
					float* deltaW;
					float* deltaB;
					float* exp;
					NeuronalNet::GPU_CUDA_allocMem(out, m_outputs * sizeof(float));
					NeuronalNet::GPU_CUDA_allocMem(deltaW, m_weightsCount * sizeof(float));
					NeuronalNet::GPU_CUDA_allocMem(deltaB, m_neuronCount * sizeof(float));
					NeuronalNet::GPU_CUDA_allocMem(exp, m_outputs * sizeof(float));

					NeuronalNet::GPU_CUDA_memset(out, 0, m_outputs * sizeof(float));
					NeuronalNet::GPU_CUDA_memset(deltaW, 0, m_weightsCount * sizeof(float));
					NeuronalNet::GPU_CUDA_memset(deltaB, 0, m_neuronCount * sizeof(float));
					NeuronalNet::GPU_CUDA_memset(exp, 0, m_outputs * sizeof(float));

					h_d_outputDifference[i] = out;
					h_d_deltaWeight[i] = deltaW;
					h_d_deltaBias[i] = deltaB;
					h_d_expected[i] = exp;
				}

				NeuronalNet::GPU_CUDA_allocMem(d_outputDifference, m_streamSize * sizeof(float*));
				NeuronalNet::GPU_CUDA_allocMem(d_deltaWeight, m_streamSize * sizeof(float*));
				NeuronalNet::GPU_CUDA_allocMem(d_deltaBias, m_streamSize * sizeof(float*));
				NeuronalNet::GPU_CUDA_allocMem(d_expected, m_streamSize * sizeof(float*));

				NeuronalNet::GPU_CUDA_transferToDevice(d_outputDifference, h_d_outputDifference, m_streamSize * sizeof(float*));
				NeuronalNet::GPU_CUDA_transferToDevice(d_deltaWeight, h_d_deltaWeight, m_streamSize * sizeof(float*));
				NeuronalNet::GPU_CUDA_transferToDevice(d_deltaBias, h_d_deltaBias, m_streamSize * sizeof(float*));
				NeuronalNet::GPU_CUDA_transferToDevice(d_expected, h_d_expected, m_streamSize * sizeof(float*));
				break;
			}
#endif
			default: {}
		}

	}
	void BackpropNet::destroyDevice()
	{
		DEBUG_FUNCTION_TIME_INTERVAL
		Net::destroyDevice();
		switch (m_hardware)
		{
#ifdef USE_CUDA
			case Hardware::gpu_cuda:
			{
				NeuronalNet::GPU_CUDA_transferToHost(d_outputDifference, h_d_outputDifference, m_streamSize * sizeof(float*));
				NeuronalNet::GPU_CUDA_transferToHost(d_deltaWeight, h_d_deltaWeight, m_streamSize * sizeof(float*));
				NeuronalNet::GPU_CUDA_transferToHost(d_deltaBias, h_d_deltaBias, m_streamSize * sizeof(float*));
				NeuronalNet::GPU_CUDA_transferToHost(d_expected, h_d_expected, m_streamSize * sizeof(float*));

				for (size_t i = 0; i < m_streamSize; ++i)
				{
					NeuronalNet::GPU_CUDA_freeMem(h_d_outputDifference[i]);
					NeuronalNet::GPU_CUDA_freeMem(h_d_deltaWeight[i]);
					NeuronalNet::GPU_CUDA_freeMem(h_d_deltaBias[i]);
					NeuronalNet::GPU_CUDA_freeMem(h_d_expected[i]);
				}

				NeuronalNet::GPU_CUDA_freeMem(d_outputDifference);
				NeuronalNet::GPU_CUDA_freeMem(d_deltaWeight);
				NeuronalNet::GPU_CUDA_freeMem(d_deltaBias);
				NeuronalNet::GPU_CUDA_freeMem(d_expected);

				delete[] h_d_outputDifference;
				delete[] h_d_deltaWeight;
				delete[] h_d_deltaBias;
				delete[] h_d_expected;

				h_d_outputDifference	= nullptr; 
				d_outputDifference		= nullptr;
				d_deltaWeight			= nullptr;
				h_d_deltaWeight			= nullptr;
				d_deltaBias				= nullptr;
				h_d_deltaBias			= nullptr;
				h_d_expected			= nullptr;
				d_expected				= nullptr;


				break;
			}
#endif
			default: {}
		}
	}


	/*inline const SignalVector& BackpropNet::internal_getError(size_t streamIndex)
	{
		switch (m_hardware)
		{
			case Hardware::gpu_cuda:
			{
				GPU_CUDA_transferToHost(m_outputDifference[streamIndex].begin(), h_d_outputDifference[streamIndex], m_outputs * sizeof(float));
				break;
			}
			default:
			{

			}
			case Hardware::cpu: {}
		}
		return m_outputDifference[streamIndex];
	}*/
	inline const MultiSignalVector& BackpropNet::internal_getError()
	{
		size_t outputNeuronBeginIndex = m_neuronCount - m_outputs - 1;
		for (size_t i = 0; i < m_expected.size(); ++i)
		{
			internal_getError(i);
		}
		return m_outputDifference;
	}
	inline const SignalVector& BackpropNet::internal_getError(size_t streamIndex)
	{
		switch (m_hardware)
		{
#ifdef USE_CUDA
			case Hardware::gpu_cuda:
			{
				//GPU_CUDA_learnBackpropagation_getOutputError(h_d_outputStream[streamIndex], m_expected[streamIndex].begin(), m_outputDifference[streamIndex].begin(), m_outputs);
				GPU_CUDA_transferToHost(m_outputDifference[streamIndex].begin(), h_d_outputDifference[streamIndex], m_outputs * sizeof(float));
				break;
			}
#endif
			case Hardware::cpu:
			{
				//size_t outputNeuronBeginIndex = m_neuronCount - m_outputs - 1;
				for (size_t y = 0; y < m_outputs; ++y)
				{
					//float netinput = m_netinputList[streamIndex][outputNeuronBeginIndex + y];

					float expected = m_expected[streamIndex][y];
					float output = m_outputStream[streamIndex][y];
					float difference = (expected - output);
					m_outputDifference[streamIndex][y] = difference;
				}
			}
			default:
			{

			}
		}
		
		return m_outputDifference[streamIndex];
	}

	void BackpropNet::CPU_learn()
	{
		DEBUG_BENCHMARK_STACK
		DEBUG_FUNCTION_TIME_INTERVAL
		//float* deltaW = new float[m_weightsCount];
		//float* deltaB = new float[m_neuronCount];

		
		
		for (size_t streamIndex = 0; streamIndex < m_streamSize; ++streamIndex)
		{
			memset(m_deltaWeight[streamIndex].begin(), 0, m_weightsCount * sizeof(float));
			memset(m_deltaBias[streamIndex].begin(), 0, m_neuronCount * sizeof(float));
			CPU_learn(streamIndex, m_deltaWeight[streamIndex].begin(), m_deltaBias[streamIndex].begin());
		}
		for (size_t streamIndex = 0; streamIndex < m_streamSize; ++streamIndex)
		{
			for (size_t w = 0; w < m_weightsCount; ++w)
			{
				m_weightsList[w] += m_deltaWeight[streamIndex][w] * m_learnParameter;
			}
			for (size_t b = 0; b < m_neuronCount; ++b)
			{
				m_biasList[b] += m_deltaBias[streamIndex][b] * m_learnParameter;
			}
		}
		//delete[] deltaW;
		//delete[] deltaB;
	}
	void BackpropNet::CPU_learn(size_t streamIndex, float* deltaWeights, float* deltaBiasList)
	{
		DEBUG_BENCHMARK_STACK
		DEBUG_FUNCTION_TIME_INTERVAL
		size_t outputNeuronBeginIndex = m_neuronCount - m_outputs;
		//size_t outputNeuronBeginIndex = m_neuronCount - m_outputs - 1;
		SignalVector outputError(m_outputs);
		//getError(streamIndex, m_expected[streamIndex]);
		internal_getError(streamIndex);
		for (size_t y = 0; y < m_outputs; ++y)
		{
			float netinput = m_netinputList[streamIndex][outputNeuronBeginIndex + y];
			float derivetive = (*m_activationDerivetiveFunc)(netinput);
			outputError[y] = derivetive * m_outputDifference[streamIndex][y];

			float deltaBias = outputError[y];
			deltaBiasList[outputNeuronBeginIndex + y] += deltaBias;
		}

		// Calculate errors for each layer:
		if (m_hiddenX > 0)
		{
			SignalVector* nextHiddenError = nullptr;


			for (long long x = m_hiddenX - 1; x >= 0; --x)
			{
				size_t hiddenNeuronBeginIndex = x * m_hiddenY;
				SignalVector* hiddenError = DBG_NEW SignalVector(m_hiddenY);

				for (size_t y = 0; y < m_hiddenY; ++y)
				{
					float sumNextLayerErrors = 0;
					size_t weightIndex = m_inputs * m_hiddenY + x * m_hiddenY * m_hiddenY + y;
					if (x == m_hiddenX - 1)
					{
						// Calculate the errorsum of the outputLayer			
						for (size_t i = 0; i < m_outputs; ++i)
						{
							sumNextLayerErrors += outputError[i] * m_weightsList[weightIndex + i * m_hiddenY];

							// Change the weight
							float deltaW = m_neuronValueList[streamIndex][hiddenNeuronBeginIndex + y] * outputError[i];
							deltaWeights[weightIndex + i * m_hiddenY] += deltaW;
						}

					}
					else
					{
						// Calculate the errorsum of the hiddenLayer
						for (size_t i = 0; i < m_hiddenY; ++i)
						{
							sumNextLayerErrors += (*nextHiddenError)[i] * m_weightsList[weightIndex + i * m_hiddenY];

							// Change the weight
							float deltaW = m_neuronValueList[streamIndex][hiddenNeuronBeginIndex + y] * (*nextHiddenError)[i];
							deltaWeights[weightIndex + i * m_hiddenY] += deltaW;
						}
					}

					(*hiddenError)[y] = (*m_activationDerivetiveFunc)(m_netinputList[streamIndex][hiddenNeuronBeginIndex + y]) *
						sumNextLayerErrors;

					float deltaBias = (*hiddenError)[y];
					deltaBiasList[x * m_hiddenY + y] += deltaBias;
				}
				if (nextHiddenError)
					delete nextHiddenError;
				nextHiddenError = hiddenError;

				if (x == 0)
				{
					// Change the last waights: inputweights
					for (size_t i = 0; i < m_hiddenY; ++i)
					{
						for (size_t y = 0; y < m_inputs; ++y)
						{
							// Change the weight
							float deltaW = m_inputStream[streamIndex][y] * (*hiddenError)[i];
							deltaWeights[y + m_inputs * i] += deltaW;
						}
					}
				}
			}
			delete nextHiddenError;
		}
		else
		{
			// Only one Layer: outputLayer

			// Change the last weights: inputweights
			
			for (size_t i = 0; i < m_outputs; ++i)
			{
				for (size_t y = 0; y < m_inputs; ++y)
				{
					// Change the weight
					float deltaW = m_inputStream[streamIndex][y] * outputError[i];
					deltaWeights[y + m_inputs * i] += deltaW;
				}
			}
		}
	}

	void BackpropNet::GPU_learn()
	{
        #ifdef USE_CUDA
		DEBUG_BENCHMARK_STACK
		DEBUG_FUNCTION_TIME_INTERVAL
		//float** d_expected;
		//float** h_d_expected = DBG_NEW float* [m_streamSize];

		{
			//Debug::DebugFuncStackTimeTrace trace("Copy Stream");
			for (size_t i = 0; i < m_streamSize; ++i)
			{
				//float* exp;
				//GPU_CUDA_allocMem(exp, m_outputs * sizeof(float));
				if (m_expectedChanged[i])
				{
					GPU_CUDA_transferToDevice(h_d_expected[i], m_expected[i].begin(), m_outputs * sizeof(float));
					m_expectedChanged[i] = false;
				}
				//h_d_expected[i] = exp;
			}
		}
		//{
		//	Debug::DebugFuncStackTimeTrace trace("GPU_CUDA_allocMem");
		//	GPU_CUDA_allocMem(d_expected, m_streamSize * sizeof(float*));
		//}
		//{
		//	Debug::DebugFuncStackTimeTrace trace("GPU_CUDA_transferToDevice");
		//	GPU_CUDA_transferToDevice(d_expected, h_d_expected, m_streamSize * sizeof(float*));
		//}
		{
			//Debug::DebugFuncStackTimeTrace trace("GPU_CUDA_learnBackpropagationStream");
			GPU_CUDA_learnBackpropagationStream(d_weightsList, d_deltaWeight, d_biasList, d_deltaBias, d_inputSignalList, d_neuronValueList, d_netinputList,
												m_inputs, m_hiddenX, m_hiddenY, m_outputs, m_neuronCount, m_weightsCount, m_activation,
												d_outputDifference, d_expected, m_learnParameter, m_streamSize);
		}
		
		//{
		//	Debug::DebugFuncStackTimeTrace trace("freeMem");
		//	for (size_t i = 0; i < m_streamSize; ++i)
		//	{
		//		GPU_CUDA_freeMem(h_d_expected[i]);
		//	}
		//	GPU_CUDA_freeMem(d_expected);
		//	delete[] h_d_expected;
		//}

		m_weightsChangedFromDeviceTraining = true;
		m_biasChangedFromDeviceTraining = true;
		//transferBiasToHost();
		//transferWeightsToHost();
    #endif
	}
	void BackpropNet::GPU_learn(size_t streamIndex)
	{
        #ifdef USE_CUDA
		DEBUG_BENCHMARK_STACK
		DEBUG_FUNCTION_TIME_INTERVAL
		float* d_expected;

		GPU_CUDA_allocMem(d_expected, m_outputs * sizeof(float));
		GPU_CUDA_transferToDevice(d_expected, m_expected[streamIndex].begin(), m_outputs * sizeof(float));
		GPU_CUDA_learnBackpropagation(d_weightsList, h_d_deltaWeight[streamIndex], d_biasList, h_d_deltaWeight[streamIndex], h_d_inputSignalList[streamIndex], h_d_neuronValueList[streamIndex], h_d_netinputList[streamIndex],
									  m_inputs, m_hiddenX, m_hiddenY, m_outputs, m_neuronCount, m_weightsCount, m_activation,
									  h_d_outputDifference[streamIndex], d_expected, m_learnParameter);

		GPU_CUDA_freeMem(d_expected);
		m_weightsChangedFromDeviceTraining = true;
		m_biasChangedFromDeviceTraining = true;
		//transferBiasToHost();
		//transferWeightsToHost();
    #endif
	}

};
