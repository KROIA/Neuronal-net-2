#pragma once

#include "net.h"
#include <vector>

class NET_API BackpropNet : public Net
{
	public:
	BackpropNet();
	~BackpropNet();

	bool build();

	void learn(const MultiSignalVector &expectedOutputVec);
	void learn(const SignalVector &expectedOutputVec);

	const SignalVector& getError();

	std::vector<float> deltaWeight;
	std::vector<float> deltaBias;
	float m_lernParameter;
	private:
	
	void CPU_learn(const MultiSignalVector& expectedOutputVec);
	void CPU_learn(const SignalVector& expectedOutputVec);

	void GPU_learn(const MultiSignalVector& expectedOutputVec);
	void GPU_learn(const SignalVector& expectedOutputVec);

	
	//static float calculateOutputError(float output)

	SignalVector m_outputDifference;
};