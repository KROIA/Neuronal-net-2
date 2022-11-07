#pragma once

#include "backend/net.h"
#include <vector>

namespace NeuronalNet
{
	class NET_API BackpropNet : public Net
	{
		public:
		BackpropNet();
		~BackpropNet();

		//virtual void setHardware(enum Hardware ware);
		bool build() override;

		void setLearnParameter(float learnParam);
		float getLearnParameter() const;

		void setExpectedOutput(const MultiSignalVector& expectedOutputVec);
		void setExpectedOutput(const SignalVector& expectedOutputVec);
		void setExpectedOutput(size_t streamIndex, const SignalVector& expectedOutputVec);
		void learn();
		void learn(size_t streamIndex);
		void learn(const MultiSignalVector& expectedOutputVec);
		void learn(const SignalVector& expectedOutputVec);
		void learn(size_t streamIndex, const SignalVector& expectedOutputVec);

		const SignalVector& getError(size_t streamIndex);
		const MultiSignalVector& getError(const MultiSignalVector& expectedOutputVec);
		const SignalVector& getError(size_t streamIndex, const SignalVector& expectedOutputVec);
		const MultiSignalVector& getError() const;

		
		protected:

		virtual void buildDevice() override;
		virtual void destroyDevice() override;

		private:
		
		

		void CPU_learn();
		void CPU_learn(size_t streamIndex, float*deltaWeights, float * deltaBiasList);

		void GPU_learn();
		void GPU_learn(size_t streamIndex);


		//static float calculateOutputError(float output)
		inline const SignalVector& internal_getError(size_t streamIndex);
		inline const MultiSignalVector& internal_getError();
		//inline const SignalVector& internal_getError(size_t streamIndex);

		MultiSignalVector m_outputDifference;
		float m_learnParameter;

		float** h_d_outputDifference;
		float** d_outputDifference;

		MultiSignalVector m_deltaWeight;
		MultiSignalVector m_deltaBias;

		float** d_deltaWeight;
		float** h_d_deltaWeight;
		float** d_deltaBias;
		float** h_d_deltaBias;


		MultiSignalVector m_expected;
		vector<bool> m_expectedChanged;
		float** d_expected;
		float** h_d_expected;
	};
	#define DEBUG_BENCHMARK_STACK
	//#define DEBUG_BENCHMARK_STACK Debug::DebugFuncStackTimeTrace trace(__PRETTY_FUNCTION__);
};
