#pragma once

#include "net.h"

class NET_API BackpropNet : public Net
{
	public:
	BackpropNet();
	~BackpropNet();

	void learn(const MultiSignalVector &expectedOutputVec);
	void learn(const SignalVector &expectedOutputVec);

	private:
	
	void CPU_learn(const MultiSignalVector& expectedOutputVec);
	void CPU_learn(const SignalVector& expectedOutputVec);

	void GPU_learn(const MultiSignalVector& expectedOutputVec);
	void GPU_learn(const SignalVector& expectedOutputVec);

	float m_lernParameter;
	//static float calculateOutputError(float output)

};