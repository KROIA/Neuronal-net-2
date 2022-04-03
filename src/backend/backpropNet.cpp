

#include "backend/backpropNet.h"

namespace NeuronalNet
{
	BackpropNet::BackpropNet()
	{
		m_learnParameter = 0.01;
		h_d_outputDifference = nullptr;
		d_outputDifference = nullptr;
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
		m_outputDifference.resize(m_streamSize,m_outputs);
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

	void BackpropNet::learn(const MultiSignalVector& expectedOutputVec)
	{
		DEBUG_FUNCTION_TIME_INTERVAL
		VERIFY_RANGE(0, expectedOutputVec.size(), m_streamSize, return)
		VERIFY_BOOL(expectedOutputVec[0].size(), m_outputs, "expectedOutputVec[0].size() != m_outputs", return)
		switch (m_hardware)
		{
			case Hardware::cpu:
			{
				CPU_learn(expectedOutputVec);
				break;
			}
			case Hardware::gpu_cuda:
			{
				GPU_learn(expectedOutputVec);
				break;
			}
		}
	}
	void BackpropNet::learn(const SignalVector& expectedOutputVec)
	{
		learn(0, expectedOutputVec);
	}
	void BackpropNet::learn(size_t streamIndex,  const SignalVector& expectedOutputVec)
	{
		DEBUG_FUNCTION_TIME_INTERVAL
		VERIFY_RANGE(0, streamIndex, m_streamSize, return)
		VERIFY_BOOL(expectedOutputVec.size(), m_outputs,"expectedOutputVec.size() != m_outputs", return)
		switch (m_hardware)
		{
			case Hardware::cpu:
			{
				CPU_learn(streamIndex,expectedOutputVec,m_weightsList,m_biasList);
				break;
			}
			case Hardware::gpu_cuda:
			{
				GPU_learn(streamIndex, expectedOutputVec);
				break;
			}
		}
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
				internal_getError(i, expectedOutputVec[i]);
			return m_outputDifference;
		}

		return internal_getError(expectedOutputVec);
	}
	const SignalVector& BackpropNet::getError(size_t streamIndex, const SignalVector& expectedOutputVec)
	{
		VERIFY_RANGE(0, streamIndex, m_streamSize - 1, return m_outputDifference[0])
		
		return internal_getError(streamIndex, expectedOutputVec);
	}
	const MultiSignalVector& BackpropNet::getError() const
	{
		switch (m_hardware)
		{
			case Hardware::gpu_cuda:
			{
				
				//GPU_CUDA_learnBackpropagation_getOutputError(h_d_outputStream[streamIndex], expectedOutputVec.begin(), m_outputDifference[streamIndex].begin(), m_outputs);
				for (size_t i = 0; i < m_streamSize; ++i)
					GPU_CUDA_transferToHost(h_d_outputDifference[i], m_outputDifference[i].begin(), m_outputs * sizeof(float));
				break;
			}
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
			case Hardware::gpu_cuda:
			{
				h_d_outputDifference = DBG_NEW float* [m_streamSize];
				for (size_t i = 0; i < m_streamSize; ++i)
				{
					float* out;
					NeuronalNet::GPU_CUDA_allocMem(out, m_outputs * sizeof(float));
					NeuronalNet::GPU_CUDA_memset(out, 0, m_outputs * sizeof(float));
					h_d_outputDifference[i] = out;
				}

				NeuronalNet::GPU_CUDA_allocMem(d_outputDifference, m_streamSize * sizeof(float*));
				NeuronalNet::GPU_CUDA_transferToDevice(d_outputDifference, h_d_outputDifference, m_streamSize * sizeof(float*));
				break;
			}
			default: {}
		}

	}
	void BackpropNet::destroyDevice()
	{
		DEBUG_FUNCTION_TIME_INTERVAL
		Net::destroyDevice();
		switch (m_hardware)
		{
			case Hardware::gpu_cuda:
			{
				NeuronalNet::GPU_CUDA_transferToHost(d_outputDifference, h_d_outputDifference, m_streamSize * sizeof(float*));

				for (size_t i = 0; i < m_streamSize; ++i)
				{
					NeuronalNet::GPU_CUDA_freeMem(h_d_outputDifference[i]);
				}

				NeuronalNet::GPU_CUDA_freeMem(d_outputDifference);
				delete[] h_d_outputDifference;

				h_d_outputDifference = nullptr; 
				d_outputDifference = nullptr;

				break;
			}
			default: {}
		}
	}


	inline const SignalVector& BackpropNet::internal_getError(size_t streamIndex)
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
	}
	inline const MultiSignalVector& BackpropNet::internal_getError(const MultiSignalVector& expectedOutputVec)
	{
		size_t outputNeuronBeginIndex = m_neuronCount - m_outputs - 1;
		for (size_t i = 0; i < expectedOutputVec.size(); ++i)
		{
			internal_getError(i, expectedOutputVec[i]);
		}
		return m_outputDifference;
	}
	inline const SignalVector& BackpropNet::internal_getError(size_t streamIndex, const SignalVector& expectedOutputVec)
	{
		switch (m_hardware)
		{
			case Hardware::gpu_cuda:
			{
				GPU_CUDA_learnBackpropagation_getOutputError(h_d_outputStream[streamIndex], expectedOutputVec.begin(), m_outputDifference[streamIndex].begin(), m_outputs);
				break;
			}
			case Hardware::cpu:
			{
				//size_t outputNeuronBeginIndex = m_neuronCount - m_outputs - 1;
				for (size_t y = 0; y < m_outputs; ++y)
				{
					//float netinput = m_netinputList[streamIndex][outputNeuronBeginIndex + y];

					float expected = expectedOutputVec[y];
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

	void BackpropNet::CPU_learn(const MultiSignalVector& expectedOutputVec)
	{
		DEBUG_FUNCTION_TIME_INTERVAL
		float* deltaW = new float[m_weightsCount];
		float* deltaB = new float[m_neuronCount];

		memset(deltaW, 0, m_weightsCount * sizeof(float));
		memset(deltaB, 0, m_neuronCount * sizeof(float));

		for (size_t streamIndex = 0; streamIndex < expectedOutputVec.size(); ++streamIndex)
		{
			CPU_learn(streamIndex, expectedOutputVec[streamIndex],
					  deltaW, deltaB);
		}

		for (size_t w = 0; w < m_weightsCount; ++w)
		{
			m_weightsList[w] += deltaW[w];
		}
		for (size_t b = 0; b < m_neuronCount; ++b)
		{
			m_biasList[b] += deltaB[b];
		}
		delete[] deltaW;
		delete[] deltaB;
	}
	void BackpropNet::CPU_learn(size_t streamIndex, const SignalVector& expectedOutputVec, float* deltaWeights, float* deltaBiasList)
	{
		DEBUG_FUNCTION_TIME_INTERVAL
		size_t outputNeuronBeginIndex = m_neuronCount - m_outputs - 1;
		SignalVector outputError(m_outputs);
		getError(streamIndex,expectedOutputVec);
		for (size_t y = 0; y < m_outputs; ++y)
		{
			float netinput = m_netinputList[streamIndex][outputNeuronBeginIndex + y];
			float derivetive = (*m_activationDerivetiveFunc)(netinput);
			outputError[y] = derivetive * m_outputDifference[streamIndex][y];

			float deltaBias = m_learnParameter * outputError[y];
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
							float deltaW = m_learnParameter * m_neuronValueList[streamIndex][hiddenNeuronBeginIndex + y] * outputError[i];
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
							float deltaW = m_learnParameter * m_neuronValueList[streamIndex][hiddenNeuronBeginIndex + y] * (*nextHiddenError)[i];
							deltaWeights[weightIndex + i * m_hiddenY] += deltaW;
						}
					}

					(*hiddenError)[y] = (*m_activationDerivetiveFunc)(m_netinputList[streamIndex][hiddenNeuronBeginIndex + y]) *
						sumNextLayerErrors;

					float deltaBias = m_learnParameter * (*hiddenError)[y];
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
							float deltaW = m_learnParameter * m_inputStream[streamIndex][y] * (*hiddenError)[i];
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
					float deltaW = m_learnParameter * m_inputStream[streamIndex][y] * outputError[i];
					deltaWeights[y + m_inputs * i] += deltaW;
				}
			}
		}
	}

	void BackpropNet::GPU_learn(const MultiSignalVector& expectedOutputVec)
	{
		DEBUG_FUNCTION_TIME_INTERVAL
		float** d_expected;
		float** h_d_expected = DBG_NEW float* [m_streamSize];

		for (size_t i = 0; i < m_streamSize; ++i)
		{
			float* exp;
			GPU_CUDA_allocMem(exp, m_outputs * sizeof(float));
			GPU_CUDA_transferToDevice(exp, expectedOutputVec[i].begin(), m_outputs * sizeof(float));
			h_d_expected[i] = exp;
		}

		GPU_CUDA_allocMem(d_expected, m_streamSize * sizeof(float*));
		GPU_CUDA_transferToDevice(d_expected, h_d_expected, m_streamSize * sizeof(float*));
		GPU_CUDA_learnBackpropagationStream(d_weightsList, d_biasList, d_inputSignalList, d_neuronValueList, d_netinputList,
											m_inputs, m_hiddenX, m_hiddenY, m_outputs, m_neuronCount, m_weightsCount, m_activation,
											d_outputDifference, d_expected, m_learnParameter, m_streamSize);
		
		for (size_t i = 0; i < m_streamSize; ++i)
		{
			GPU_CUDA_freeMem(h_d_expected[i]);
		}
		GPU_CUDA_freeMem(d_expected);
		delete[] h_d_expected;

	}
	void BackpropNet::GPU_learn(size_t streamIndex, const SignalVector& expectedOutputVec)
	{
		DEBUG_FUNCTION_TIME_INTERVAL
		float* d_expected;

		GPU_CUDA_allocMem(d_expected, m_outputs * sizeof(float));
		GPU_CUDA_transferToDevice(d_expected, expectedOutputVec.begin(), m_outputs * sizeof(float));
		GPU_CUDA_learnBackpropagation(d_weightsList, d_biasList, d_inputSignalList[streamIndex], d_neuronValueList[streamIndex], d_netinputList[streamIndex],
									  m_inputs, m_hiddenX, m_hiddenY, m_outputs, m_neuronCount, m_weightsCount, m_activation,
									  d_outputDifference[streamIndex], d_expected, m_learnParameter);

		GPU_CUDA_freeMem(d_expected);
	}

};