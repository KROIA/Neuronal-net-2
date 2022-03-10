

#include "backend/backpropNet.h"

namespace NeuronalNet
{
	BackpropNet::BackpropNet()
	{
		m_lernParameter = 0.01;
	}
	BackpropNet::~BackpropNet()
	{

	}

	bool BackpropNet::build()
	{
		bool ret = Net::build();
		m_outputDifference.resize(m_outputs);

		for (size_t y = 0; y < m_neuronCount; ++y)
		{
			m_biasList[y] = 0;
		}

		return ret;
	}

	void BackpropNet::learn(const MultiSignalVector& expectedOutputVec)
	{
		DEBUG_FUNCTION_TIME_INTERVAL
	}
	void BackpropNet::learn(const SignalVector& expectedOutputVec)
	{
		DEBUG_FUNCTION_TIME_INTERVAL
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
	const SignalVector& BackpropNet::getError()
	{
		return m_outputDifference;
	}

	void BackpropNet::CPU_learn(const MultiSignalVector& expectedOutputVec)
	{
		DEBUG_FUNCTION_TIME_INTERVAL
	}
	void BackpropNet::CPU_learn(const SignalVector& expectedOutputVec)
	{
		DEBUG_FUNCTION_TIME_INTERVAL
			deltaWeight.clear();
		deltaBias.clear();
		// Calculate all output Errors:
		//SignalVector outputError(m_outputs);
		size_t outputNeuronBeginIndex = m_neuronCount - m_outputs - 1;
		SignalVector outputError(m_outputs);
		for (size_t y = 0; y < m_outputs; ++y)
		{
			float netinput = m_netinputList[0][outputNeuronBeginIndex + y];
			float derivetive = (*m_activationDerivetiveFunc)(netinput);
			float expected = expectedOutputVec[y];
			float output = m_outputStream[0][y];
			float difference = (expected - output);
			m_outputDifference[y] = difference;
			outputError[y] = derivetive * difference;


			float deltaBias = m_lernParameter * outputError[y];
			m_biasList[outputNeuronBeginIndex + y] += deltaBias;
			this->deltaBias.push_back(deltaBias);
		}

		// Calculate errors for each layer:
		if (m_hiddenX > 0)
		{
			SignalVector* prevHiddenError = nullptr;


			for (long long x = m_hiddenX - 1; x >= 0; --x)
			{
				size_t hiddenNeuronBeginIndex = x * m_hiddenY;
				SignalVector* hiddenError = new SignalVector(m_hiddenY);

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
							float deltaW = m_lernParameter * m_neuronValueList[0][hiddenNeuronBeginIndex + y] * outputError[i];
							m_weightsList[weightIndex + i * m_hiddenY] += deltaW;
							deltaWeight.push_back(deltaW);
						}

					}
					else
					{
						// Calculate the errorsum of the hiddenLayer
						for (size_t i = 0; i < m_hiddenY; ++i)
						{
							sumNextLayerErrors += (*prevHiddenError)[i] * m_weightsList[weightIndex + i * m_hiddenY];

							// Change the weight
							float deltaW = m_lernParameter * m_neuronValueList[0][hiddenNeuronBeginIndex + y] * (*prevHiddenError)[i];
							m_weightsList[weightIndex + i * m_hiddenY] += deltaW;
							deltaWeight.push_back(deltaW);
						}
					}

					(*hiddenError)[y] = (*m_activationDerivetiveFunc)(m_netinputList[0][hiddenNeuronBeginIndex + y]) *
						sumNextLayerErrors;

					float deltaBias = m_lernParameter * (*hiddenError)[y];
					m_biasList[x * m_hiddenY + y] += deltaBias;
					this->deltaBias.push_back(deltaBias);

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
							float deltaW = m_lernParameter * m_inputStream[0][y] * (*hiddenError)[i];
							m_weightsList[y * m_hiddenY + i] += deltaW;
							deltaWeight.push_back(deltaW);
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
					float deltaW = m_lernParameter * m_inputStream[0][y] * outputError[i];
					m_weightsList[y * m_outputs + i] += deltaW;
					deltaWeight.push_back(deltaW);
				}
			}
		}
	}

	void BackpropNet::GPU_learn(const MultiSignalVector& expectedOutputVec)
	{
		DEBUG_FUNCTION_TIME_INTERVAL
	}
	void BackpropNet::GPU_learn(const SignalVector& expectedOutputVec)
	{
		DEBUG_FUNCTION_TIME_INTERVAL
	}

};