#pragma once

#include "net.h"

class NET_API BackpropNet : public Net
{
	public:
	BackpropNet();
	~BackpropNet();

	void learn(MultiSignalVector expectedOutputVec);
	void learn(SignalVector expectedOutputVec);

	private:
	//static float calculateOutputError(float output)

};