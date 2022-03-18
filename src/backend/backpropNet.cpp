

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
		m_outputDifference.resize(m_streamSize,m_outputs);

		/*for (size_t y = 0; y < m_neuronCount; ++y)
		{
			m_biasList[y] = 0;
		}*/

		return ret;
	}

	void BackpropNet::learn(const MultiSignalVector& expectedOutputVec)
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
	void BackpropNet::learn(const SignalVector& expectedOutputVec)
	{
		learn(0, expectedOutputVec);
	}
	void BackpropNet::learn(size_t streamIndex,  const SignalVector& expectedOutputVec)
	{
		DEBUG_FUNCTION_TIME_INTERVAL
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
		/*
		DEBUG_FUNCTION_TIME_INTERVAL
		for(size_t i=0; i<expectedOutputVec.size(); ++i)
			CPU_learn(i,expectedOutputVec[i]);
		*/	
		DEBUG_FUNCTION_TIME_INTERVAL
		//VERIFY_RANGE(0, streamIndex, m_streamSize, return)

		deltaWeight.clear();
		deltaBias.clear();
		// Calculate all output Errors:
		//SignalVector outputError(m_outputs);
		vector<float> deltaWList(m_weightsCount, 0);
		vector<float> deltaBList(m_neuronCount, 0);
		size_t outputNeuronBeginIndex = m_neuronCount - m_outputs - 1;
		SignalVector outputError(m_outputs);

		getError(expectedOutputVec);
		for (size_t streamIndex = 0; streamIndex < expectedOutputVec.size(); ++streamIndex)
		{
			for (size_t y = 0; y < m_outputs; ++y)
			{
				float netinput = m_netinputList[streamIndex][outputNeuronBeginIndex + y];
				float derivetive = (*m_activationDerivetiveFunc)(netinput);
				//float expected = expectedOutputVec[streamIndex][y];
				//float output = m_outputStream[streamIndex][y];
				//float difference = (expected - output);
				//m_outputDifference[streamIndex][y] = difference;
				outputError[y] = derivetive * m_outputDifference[streamIndex][y];


				float deltaBias = m_lernParameter * outputError[y];
				deltaBList[outputNeuronBeginIndex + y] += deltaBias;
				//m_biasList[outputNeuronBeginIndex + y] += deltaBias;
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
								float deltaW = m_lernParameter * m_neuronValueList[streamIndex][hiddenNeuronBeginIndex + y] * outputError[i];
								deltaWList[weightIndex + i * m_hiddenY] += deltaW;
								//m_weightsList[weightIndex + i * m_hiddenY] += deltaW;
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
								float deltaW = m_lernParameter * m_neuronValueList[streamIndex][hiddenNeuronBeginIndex + y] * (*prevHiddenError)[i];
								deltaWList[weightIndex + i * m_hiddenY] += deltaW;
								//m_weightsList[weightIndex + i * m_hiddenY] += deltaW;
								deltaWeight.push_back(deltaW);
							}
						}

						(*hiddenError)[y] = (*m_activationDerivetiveFunc)(m_netinputList[streamIndex][hiddenNeuronBeginIndex + y]) *
							sumNextLayerErrors;

						float deltaBias = m_lernParameter * (*hiddenError)[y];
						//m_biasList[x * m_hiddenY + y] += deltaBias;
						deltaBList[x * m_hiddenY + y] += deltaBias;
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
								float deltaW = m_lernParameter * m_inputStream[streamIndex][y] * (*hiddenError)[i];
								deltaWList[y * m_hiddenY + i] += deltaW;
								//m_weightsList[y * m_hiddenY + i] += deltaW;
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
						float deltaW = m_lernParameter * m_inputStream[streamIndex][y] * outputError[i];
						deltaWList[y * m_outputs + i] += deltaW;
						//m_weightsList[y * m_outputs + i] += deltaW;
						deltaWeight.push_back(deltaW);
					}
				}
			}
		}
		for (size_t i = 0; i < m_weightsCount; ++i)
			m_weightsList[i] += deltaWList[i];
		for (size_t i = 0; i < m_neuronCount; ++i)
			m_biasList[i]    += deltaBList[i];
	}
	void BackpropNet::CPU_learn(size_t streamIndex, const SignalVector& expectedOutputVec)
	{
		DEBUG_FUNCTION_TIME_INTERVAL
		VERIFY_RANGE(0, streamIndex,m_streamSize,return)

		deltaWeight.clear();
		deltaBias.clear();
		// Calculate all output Errors:
		//SignalVector outputError(m_outputs);
		size_t outputNeuronBeginIndex = m_neuronCount - m_outputs - 1;
		SignalVector outputError(m_outputs);
		getError(streamIndex,expectedOutputVec);
		for (size_t y = 0; y < m_outputs; ++y)
		{
			float netinput = m_netinputList[streamIndex][outputNeuronBeginIndex + y];
			float derivetive = (*m_activationDerivetiveFunc)(netinput);
			//float expected = expectedOutputVec[streamIndex][y];
			//float output = m_outputStream[streamIndex][y];
			//float difference = (expected - output);
			//m_outputDifference[streamIndex][y] = difference;
			outputError[y] = derivetive * m_outputDifference[streamIndex][y];


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
							float deltaW = m_lernParameter * m_neuronValueList[streamIndex][hiddenNeuronBeginIndex + y] * outputError[i];
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
							float deltaW = m_lernParameter * m_neuronValueList[streamIndex][hiddenNeuronBeginIndex + y] * (*prevHiddenError)[i];
							m_weightsList[weightIndex + i * m_hiddenY] += deltaW;
							deltaWeight.push_back(deltaW);
						}
					}

					(*hiddenError)[y] = (*m_activationDerivetiveFunc)(m_netinputList[streamIndex][hiddenNeuronBeginIndex + y]) *
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
							float deltaW = m_lernParameter * m_inputStream[streamIndex][y] * (*hiddenError)[i];
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
					float deltaW = m_lernParameter * m_inputStream[streamIndex][y] * outputError[i];
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
	void BackpropNet::GPU_learn(size_t streamIndex, const SignalVector& expectedOutputVec)
	{
		DEBUG_FUNCTION_TIME_INTERVAL
	}

};