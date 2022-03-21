

#include "backend/backpropNet.h"

namespace NeuronalNet
{
	BackpropNet::BackpropNet()
	{
		m_learnParameter = 0.01;
	}
	BackpropNet::~BackpropNet()
	{
		//deltaWeight.clear();
		//deltaBias.clear();
	}

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
		switch (m_hardware)
		{
			case Hardware::cpu:
			{
				CPU_learn(expectedOutputVec);
				break;
			}
			case Hardware::gpu_cuda:
			{

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
		switch (m_hardware)
		{
			case Hardware::cpu:
			{
				CPU_learn(streamIndex,expectedOutputVec);
				break;
			}
			case Hardware::gpu_cuda:
			{

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
		if (expectedOutputVec.size() > m_streamSize)
		{
			CONSOLE_FUNCTION("Warning: expectedOutputVec.size() > m_streamSize")
			for (size_t i = 0; i < m_streamSize; ++i)
				internal_getError(i,expectedOutputVec[i]);
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
		return m_outputDifference;
	}

	inline const SignalVector& BackpropNet::internal_getError(size_t streamIndex)
	{
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
		size_t outputNeuronBeginIndex = m_neuronCount - m_outputs - 1;
		for (size_t y = 0; y < m_outputs; ++y)
		{
			float netinput = m_netinputList[streamIndex][outputNeuronBeginIndex + y];

			float expected = expectedOutputVec[y];
			float output = m_outputStream[streamIndex][y];
			float difference = (expected - output);
			m_outputDifference[streamIndex][y] = difference;
		}
		return m_outputDifference[streamIndex];
	}

	void BackpropNet::CPU_learn(const MultiSignalVector& expectedOutputVec)
	{
		DEBUG_FUNCTION_TIME_INTERVAL

		for (size_t streamIndex = 0; streamIndex < expectedOutputVec.size(); ++streamIndex)
		{
			CPU_learn(streamIndex, expectedOutputVec[streamIndex]);
		}
	}
	void BackpropNet::CPU_learn(size_t streamIndex, const SignalVector& expectedOutputVec)
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
			m_biasList[outputNeuronBeginIndex + y] += deltaBias;
		}

		// Calculate errors for each layer:
		if (m_hiddenX > 0)
		{
			SignalVector* prevHiddenError = nullptr;


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
							m_weightsList[weightIndex + i * m_hiddenY] += deltaW;
						}

					}
					else
					{
						// Calculate the errorsum of the hiddenLayer
						for (size_t i = 0; i < m_hiddenY; ++i)
						{
							sumNextLayerErrors += (*prevHiddenError)[i] * m_weightsList[weightIndex + i * m_hiddenY];

							// Change the weight
							float deltaW = m_learnParameter * m_neuronValueList[streamIndex][hiddenNeuronBeginIndex + y] * (*prevHiddenError)[i];
							m_weightsList[weightIndex + i * m_hiddenY] += deltaW;
						}
					}

					(*hiddenError)[y] = (*m_activationDerivetiveFunc)(m_netinputList[streamIndex][hiddenNeuronBeginIndex + y]) *
						sumNextLayerErrors;

					float deltaBias = m_learnParameter * (*hiddenError)[y];
					m_biasList[x * m_hiddenY + y] += deltaBias;
				}
				if (prevHiddenError)
					delete prevHiddenError;
				prevHiddenError = hiddenError;

				if (x == 0)
				{
					// Change the last waights: inputweights
					for (size_t y = 0; y < m_inputs; ++y)
					{
						for (size_t i = 0; i < m_hiddenY; ++i)
						{
							// Change the weight
							float deltaW = m_learnParameter * m_inputStream[streamIndex][y] * (*hiddenError)[i];
							m_weightsList[y * m_hiddenY + i] += deltaW;
						}
					}
				}
			}
			delete prevHiddenError;
		}
		else
		{
			// Only one Layer: outputLayer

			// Change the last waights: inputweights
			for (size_t y = 0; y < m_inputs; ++y)
			{
				for (size_t i = 0; i < m_outputs; ++i)
				{
					// Change the weight
					float deltaW = m_learnParameter * m_inputStream[streamIndex][y] * outputError[i];
					m_weightsList[y * m_outputs + i] += deltaW;
				}
			}
		}
	}

	void BackpropNet::GPU_learn(const MultiSignalVector& expectedOutputVec)
	{
		DEBUG_FUNCTION_TIME_INTERVAL
	}
	void BackpropNet::GPU_learn(size_t streamIndex, const SignalVector& expectedOutputVec)
	{
		DEBUG_FUNCTION_TIME_INTERVAL
	}

};