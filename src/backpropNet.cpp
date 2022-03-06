

#include "backpropNet.h"

BackpropNet::BackpropNet()
{

}
BackpropNet::~BackpropNet()
{

}

void BackpropNet::learn(const MultiSignalVector &expectedOutputVec)
{

}
void BackpropNet::learn(const SignalVector &expectedOutputVec) 
{
	
	
}

void BackpropNet::CPU_learn(const MultiSignalVector& expectedOutputVec)
{

}
void BackpropNet::CPU_learn(const SignalVector& expectedOutputVec)
{

	// Calculate all output Errors:
	SignalVector outputError(m_outputs);
	size_t outputNeuronBeginIndex = m_neuronCount - m_outputs;
	for (size_t y = 0; y < m_outputs; ++y)
	{
		outputError[y] = (*m_activationDerivetiveFunc)(m_netinputList[0][outputNeuronBeginIndex + y]) *
			(expectedOutputVec[y] - m_outputStream[0][y]);
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
				size_t weightIndex = (x + 1) * m_hiddenY + y;
				if (x == m_hiddenX - 1)
				{
					// Calculate the errorsum of the outputLayer			
					for (size_t i = 0; i < m_outputs; ++i)
					{
						sumNextLayerErrors += outputError[i] * m_weightsList[weightIndex + i * m_hiddenY];

						// Change the weight
						float deltaW = m_lernParameter * m_neuronValueList[0][x * m_hiddenY + y] * outputError[i];
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
						float deltaW = m_lernParameter * m_neuronValueList[0][x * m_hiddenY + y] * (*hiddenError)[i];
						m_weightsList[weightIndex + i * m_hiddenY] += deltaW;
					}
				}

				(*hiddenError)[y] = (*m_activationDerivetiveFunc)(m_netinputList[0][outputNeuronBeginIndex + y]) *
					sumNextLayerErrors;


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
						m_weightsList[y + i * m_hiddenY] += deltaW;
					}
				}
			}
		}
	}
	else
	{
		// Only one Layer: outputLayer
	}
}

void BackpropNet::GPU_learn(const MultiSignalVector& expectedOutputVec)
{

}
void BackpropNet::GPU_learn(const SignalVector& expectedOutputVec)
{

}