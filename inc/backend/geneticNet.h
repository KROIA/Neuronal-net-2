#pragma once

#include "net.h"
#include <vector>

namespace NeuronalNet
{
	class NET_API GeneticNet
	{
		public:
		GeneticNet(size_t netCount);
		~GeneticNet();

		// Net interface
		void setDimensions(size_t inputs, size_t hiddenX, size_t hiddenY, size_t outputs);
		void setStreamSize(size_t size);
		size_t getStreamSize() const;
		size_t getInputCount() const;
		size_t getHiddenXCount() const;
		size_t getHiddenYCount() const;
		size_t getOutputCount() const;
		size_t getNeuronCount() const;

		void setActivation(Activation act);
		Activation getActivation() const;

		void setHardware(enum Hardware ware);
		Hardware getHardware() const;

		void enableBias(bool enable);
		bool isBiasEnabled() const;

		bool build();
		bool isBuilt() const;
		void randomizeWeights();
	    bool randomizeWeights(size_t from, size_t to);
		void randomizeBias();

		void setInputVector(size_t netInedex, float* signalList);
		void setInputVector(size_t netInedex, size_t stream, float* signalList);
		void setInputVector(size_t netInedex, const SignalVector& signalList);
		void setInputVector(size_t netInedex, size_t stream, const SignalVector& signalList);
		void setInputVector(size_t netInedex, const MultiSignalVector& streamVector);

		void setInput(size_t netInedex, size_t input, float signal);
		void setInput(size_t netInedex, size_t stream, size_t input, float signal);
		float getInput(size_t netInedex,  size_t input) const;
		float getInput(size_t netInedex,  size_t stream, size_t input) const;
		const SignalVector& getInputVector(size_t netInedex,  size_t stream = 0);
		const MultiSignalVector& getInputStreamVector(size_t netInedex);
		const SignalVector& getOutputVector(size_t netInedex, size_t stream = 0);
		const MultiSignalVector& getOutputStreamVector(size_t netInedex);

		MultiSignalVector getNetinputStreamVector(size_t netInedex) const;
		MultiSignalVector getNeuronValueStreamVector(size_t netInedex) const;

		void setWeight(size_t netInedex, size_t layer, size_t neuron, size_t input, float weight);
		void setWeight(size_t netInedex, const std::vector<float>& list);
		void setWeight(size_t netInedex, const float* list);
		void setWeight(size_t netInedex, const float* list, size_t to);
		void setWeight(size_t netInedex, const float* list, size_t insertOffset, size_t count);
		float getWeight(size_t netInedex, size_t layer, size_t neuron, size_t input) const;
		const float* getWeight(size_t netInedex) const;
		size_t getWeightSize() const;
		const float* getBias(size_t netInedex) const;



		Net* getNet(size_t index) const;
		void setMutationChance(float chance); // 0 no chance, 1 every time
		float getMutatuionChance() const;
		void setMutationFactor(float radius); // a +- value for the min max range of random mutation. w = deltaW + oldW
		float getMutationFactor() const;
		void learn(const std::vector<float> &ranks); // Ranks must be positive otherwise they will be set to 0



		private:
		void initiateNets(size_t netCount);

		void CPU_learn(std::vector<float> ranks);
		void CPU_learn_createNewPair(const float* wOldFirst,
								     const float* wOldSecond,
								     float* wNewFirst,
								     float* wNewSecond,
									 size_t count);
		void CPU_learn_crossOver(const float* wOldFirst,
								 const float* wOldSecond,
								 float* wNewFirst,
								 float* wNewSecond,
							     size_t count,
								 size_t crossoverPoint);
		void CPU_learn_mutate(float* weights, size_t count,
						      float mutChance, float mutFac);
		


		std::vector<Net*> m_nets;
		float m_mutationFactor;
		float m_mutationChance;

		size_t m_randomSeed;
	};

#define PRINT_ERROR_NET_INDEX_OUTOFRANGE(index) \
	PRINT_ERROR("Net index out of range: index = " << index << " min = 0, max = " << m_nets.size() - 1);

#define FOR_EVERY_NET(func) \
	for(size_t i=0; i<m_nets.size(); ++i) \
		m_nets[i]->func;
}