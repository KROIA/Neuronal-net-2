#pragma once

#include "net.h"
#include <vector>

namespace NeuronalNet
{
	class NET_API BackpropNet : public Net
	{
		public:
		BackpropNet();
		~BackpropNet();

		bool build();

		void learn(const MultiSignalVector& expectedOutputVec);
		void learn(const SignalVector& expectedOutputVec);
		void learn(size_t streamIndex, const SignalVector& expectedOutputVec);

		const SignalVector& getError(size_t streamIndex);
		const MultiSignalVector& getError(const MultiSignalVector& expectedOutputVec);
		const SignalVector& getError(size_t streamIndex, const SignalVector& expectedOutputVec);
		const MultiSignalVector& getError() const;

		std::vector<float> deltaWeight;
		std::vector<float> deltaBias;
		float m_lernParameter;
		private:

		void CPU_learn(const MultiSignalVector& expectedOutputVec);
		void CPU_learn(size_t streamIndex, const SignalVector& expectedOutputVec);

		void GPU_learn(const MultiSignalVector& expectedOutputVec);
		void GPU_learn(size_t streamIndex, const SignalVector& expectedOutputVec);


		//static float calculateOutputError(float output)
		inline const SignalVector& internal_getError(size_t streamIndex);
		inline const MultiSignalVector& internal_getError(const MultiSignalVector& expectedOutputVec);
		inline const SignalVector& internal_getError(size_t streamIndex, const SignalVector& expectedOutputVec);

		MultiSignalVector m_outputDifference;
	};
};