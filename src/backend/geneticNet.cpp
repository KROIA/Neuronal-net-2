#include "backend/geneticNet.h"
#ifdef USE_CUDA
#include "net_kernel.cuh"
#endif
#include <algorithm>

namespace NeuronalNet
{
	GeneticNet::GeneticNet(size_t netCount)
	{
		if (netCount < 2)
			netCount = 2;

		initiateNets(netCount);
	}
	GeneticNet::~GeneticNet()
	{

	}

	size_t GeneticNet::getNetCount() const
	{
		return m_nets.size();
	}

	void GeneticNet::setDimensions(size_t inputs, size_t hiddenX, size_t hiddenY, size_t outputs)
	{
		FOR_EVERY_NET(setDimensions(inputs, hiddenX, hiddenY, outputs));
	}
	void GeneticNet::setStreamSize(size_t size)
	{
		FOR_EVERY_NET(setStreamSize(size));
	}
	size_t GeneticNet::getStreamSize() const
	{
		return m_nets[0]->getStreamSize();
	}
	size_t GeneticNet::getInputCount() const
	{
		return m_nets[0]->getInputCount();
	}
	size_t GeneticNet::getHiddenXCount() const
	{
		return m_nets[0]->getHiddenXCount();
	}
	size_t GeneticNet::getHiddenYCount() const
	{
		return m_nets[0]->getHiddenYCount();
	}
	size_t GeneticNet::getOutputCount() const
	{
		return m_nets[0]->getOutputCount();
	}
	size_t GeneticNet::getNeuronCount() const
	{
		return m_nets[0]->getNeuronCount();
	}

	void GeneticNet::setActivation(Activation act)
	{
		FOR_EVERY_NET(setActivation(act));
	}
	Activation GeneticNet::getActivation() const
	{
		return m_nets[0]->getActivation();
	}

	void GeneticNet::setHardware(enum Hardware ware)
	{
		FOR_EVERY_NET(setHardware(ware));
	}
	Hardware GeneticNet::getHardware() const
	{
		return m_nets[0]->getHardware();
	}

	void GeneticNet::enableBias(bool enable)
	{
		FOR_EVERY_NET(enableBias(enable));
	}
	bool GeneticNet::isBiasEnabled() const
	{
		return m_nets[0]->isBiasEnabled();
	}

	bool GeneticNet::build()
	{
		bool success = m_nets[0]->getInputCount() > 0 && m_nets[0]->getOutputCount() > 0;
		FOR_EVERY_NET(build());
		return success;
	}
	bool GeneticNet::isBuilt() const
	{
		return m_nets[0]->isBuilt();
	}
	void GeneticNet::randomizeWeights()
	{
		FOR_EVERY_NET(randomizeWeights());
	}
	bool GeneticNet::randomizeWeights(size_t from, size_t to)
	{
		bool success = true;
		for (size_t i = 0; i < m_nets.size(); ++i) 
			success &= m_nets[i]->randomizeWeights(from, to);
		return success;
	}
	void GeneticNet::randomizeBias()
	{
		FOR_EVERY_NET(randomizeBias());
	}

	void GeneticNet::setInputVector(size_t netIndex, float* signalList)
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return;
		}
		n->setInputVector(signalList);
	}
	void GeneticNet::setInputVector(size_t netIndex, size_t stream, float* signalList)
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return;
		}
		n->setInputVector(stream, signalList);
	}
	void GeneticNet::setInputVector(size_t netIndex, const SignalVector& signalList)
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return;
		}
		n->setInputVector(signalList);
	}
	void GeneticNet::setInputVector(size_t netIndex, size_t stream, const SignalVector& signalList)
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return;
		}
		n->setInputVector(stream, signalList);
	}
	void GeneticNet::setInputVector(size_t netIndex, const MultiSignalVector& streamVector)
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return;
		}
		n->setInputVector(streamVector);
	}

	void GeneticNet::setInput(size_t netIndex, size_t input, float signal)
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return;
		}
		n->setInput(input, signal);
	}
	void GeneticNet::setInput(size_t netIndex, size_t stream, size_t input, float signal)
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return;
		}
		n->setInput(stream, input, signal);
	}
	float GeneticNet::getInput(size_t netIndex, size_t input) const
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return 0;
		}
		return n->getInput(input);
	}
	float GeneticNet::getInput(size_t netIndex, size_t stream, size_t input) const
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return 0;
		}
		return n->getInput(stream, input);
	}
	const SignalVector& GeneticNet::getInputVector(size_t netIndex, size_t stream)
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return 0;
		}
		return n->getInputVector(stream);
	}
	const MultiSignalVector& GeneticNet::getInputStreamVector(size_t netIndex)
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			const static MultiSignalVector dummy;
			return dummy;
		}
		return n->getInputStreamVector();
	}
	const SignalVector& GeneticNet::getOutputVector(size_t netIndex, size_t stream)
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			const static SignalVector dummy;
			return dummy;
		}
		return n->getOutputVector(stream);
	}
	const MultiSignalVector& GeneticNet::getOutputStreamVector(size_t netIndex)
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			const static MultiSignalVector dummy;
			return dummy;
		}
		return n->getOutputStreamVector();
	}
	float GeneticNet::getOutput(size_t netIndex, size_t output)
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return 0;
		}
		return n->getOutput(output);
	}
	float GeneticNet::getOutput(size_t netIndex, size_t stream, size_t output)
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return 0;
		}
		return n->getOutput(stream, output);
	}

	MultiSignalVector GeneticNet::getNetinputStreamVector(size_t netIndex) const
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			const static MultiSignalVector dummy;
			return dummy;
		}
		return n->getNetinputStreamVector();
	}
	MultiSignalVector GeneticNet::getNeuronValueStreamVector(size_t netIndex) const
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			const static MultiSignalVector dummy;
			return dummy;
		}
		return n->getNeuronValueStreamVector();
	}

	void GeneticNet::setWeight(size_t netIndex, size_t layer, size_t neuron, size_t input, float weight)
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return;
		}
		n->setWeight(layer, neuron, input, weight);
	}
	void GeneticNet::setWeight(size_t netIndex, const std::vector<float>& list)
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return;
		}
		n->setWeight(list);
	}
	void GeneticNet::setWeight(size_t netIndex, const float* list)
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return;
		}
		n->setWeight(list);
	}
	void GeneticNet::setWeight(size_t netIndex, const float* list, size_t to)
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return;
		}
		n->setWeight(list, to);
	}
	void GeneticNet::setWeight(size_t netIndex, const float* list, size_t insertOffset, size_t count)
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return;
		}
		n->setWeight(list, insertOffset, count);
	}
	float GeneticNet::getWeight(size_t netIndex, size_t layer, size_t neuron, size_t input) const
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return 0;
		}
		return n->getWeight(layer, neuron, input);
	}
	const float* GeneticNet::getWeight(size_t netIndex) const
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return nullptr;
		}
		return n->getWeight();
	}
	size_t GeneticNet::getWeightSize() const
	{
		return m_nets[0]->getWeightSize();
	}
	void GeneticNet::setBias(size_t netIndex, size_t layer, size_t neuron, float bias)
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return;
		}
		m_nets[netIndex]->setBias(layer, neuron, bias);
	}
	void GeneticNet::setBias(size_t netIndex, float* list)
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return;
		}
		m_nets[netIndex]->setBias(list);
	}
	float GeneticNet::getBias(size_t netIndex, size_t layer, size_t neuron)
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return 0;
		}
		return m_nets[netIndex]->getBias(layer, neuron);
	}
	const float* GeneticNet::getBias(size_t netIndex) const
	{
		Net* n = getNet(netIndex);
		if (!n)
		{
			PRINT_ERROR_NET_INDEX_OUTOFRANGE(netIndex);
			return nullptr;
		}
		return n->getBias();
	}



	Net* GeneticNet::getNet(size_t index) const
	{
        if (index >= m_nets.size())
			return nullptr;
		return m_nets[index];
	}

	void GeneticNet::setMutationChance(float chance)
	{
		m_mutationChance = chance;
		if (m_mutationChance < 0)
			m_mutationChance = 0;
		if (m_mutationChance > 1)
			m_mutationChance = 1;
	}
	float GeneticNet::getMutatuionChance() const
	{
		return m_mutationChance;
	}
	void GeneticNet::setMutationFactor(float radius)
	{
		m_mutationFactor = radius;
		if (m_mutationFactor < 0)
			m_mutationFactor = -m_mutationFactor;
	}
	float GeneticNet::getMutationFactor() const
	{
		return m_mutationFactor;
	}
	void GeneticNet::calculate()
	{
		for (size_t i = 0; i < m_nets.size(); ++i)
		{
			m_nets[i]->calculate();
		}
	}
	void GeneticNet::learn(const std::vector<float> &ranks)
	{
		if (ranks.size() != m_nets.size())
		{
			PRINT_ERROR("(ranks.size() == " << ranks.size() << ") != (netCount == " << m_nets.size());
			return;
		}

		

		switch (getHardware())
		{
			case Hardware::cpu:
			{
				CPU_learn(ranks);
				break;
			}
#ifdef USE_CUDA
			case Hardware::gpu_cuda:
			{

				break;
			}
#endif
			default:
			{
				CONSOLE_FUNCTION("Specific hardware: "<<getHardware()<<" not implemented for the genetic algorithm");
			}
		}
	}
	
	void GeneticNet::initiateNets(size_t netCount)
	{
        m_nets.reserve(netCount);
        for(size_t i=0; i<netCount; ++i)
        {
            m_nets.push_back(new Net());
        }
	}

	void GeneticNet::CPU_learn(std::vector<float> ranks)
	{
		// Sum the ranks
		float rankSum = 0;
        float min = 99999, max = -99999;
		for (size_t i = 0; i < ranks.size(); ++i)
		{
            if (ranks[i] < min)
                min = ranks[i];
            else if(ranks[i] > max)
                max = ranks[i];
		}
        if(min < 0)
        {
            for (size_t i = 0; i < ranks.size(); ++i)
            {
                ranks[i] -= min;
            }
        }

        for (size_t i = 0; i < ranks.size(); ++i)
        {
             rankSum += ranks[i];
        }

		// Sorting ranks
		struct RankData
		{
			float rank;
			size_t netIndex;
		};
		std::vector<RankData> netRanks;
		netRanks.reserve(m_nets.size());
		for (size_t i = 0; i < m_nets.size(); ++i)
			netRanks.push_back(RankData{ ranks[i], i });
		std::sort(netRanks.begin(), netRanks.end(), [](const RankData& lhs, const RankData& rhs)
		{
			return lhs.rank > rhs.rank; // Descending  order
		});

		// Preparate rankings
		float rankBegin = 0;
		float** newWeights = new float* [netRanks.size()+1];
		size_t weightCount = getWeightSize();
		for (size_t i = 0; i < netRanks.size(); ++i)
		{
			rankBegin += netRanks[i].rank;
			netRanks[i].rank = rankBegin;
			

			newWeights[i] = new float[weightCount];
		}
		newWeights[netRanks.size()] = new float[weightCount];

		size_t currentNewWeightIndex = 0;
		size_t pairs = m_nets.size() / 2 + m_nets.size() % 2;
		for (size_t i = 0; i < pairs; ++i)
		{
			// Selection
			float randVal1 = Net::getRandomValue(0, rankSum);
			

			size_t netIndex1 = 0;
			size_t netIndex2 = 0;
			for (size_t j = 0; j < m_nets.size(); ++j)
			{
				if (netRanks[j].rank > randVal1)
					netIndex1 = netRanks[j].netIndex;
			}
            size_t tryCount = 10;
			do {
				float randVal2 = Net::getRandomValue(0, rankSum);
				for (size_t j = 0; j < m_nets.size(); ++j)
				{
					if (netRanks[j].rank > randVal2)
						netIndex2 = netRanks[j].netIndex;
				}
                --tryCount;
            } while (netIndex2 == netIndex1 && tryCount != 0);

            /*if(tryCount == 0)
                continue;*/
			CPU_learn_createNewPair(m_nets[netIndex1]->getWeight(),
									m_nets[netIndex2]->getWeight(),
									newWeights[currentNewWeightIndex],
									newWeights[currentNewWeightIndex + 1],
									weightCount);
			currentNewWeightIndex += 2;
		}

		for (size_t i = 0; i < m_nets.size(); ++i)
		{
			m_nets[i]->setWeight(newWeights[i]);
			delete[] newWeights[i];
		}
		delete[] newWeights[netRanks.size()];
		delete[] newWeights;
	}
	void GeneticNet::CPU_learn_createNewPair(const float* wOldFirst,
											 const float* wOldSecond,
										     float* wNewFirst,
										     float* wNewSecond,
											 size_t count)
	{
		size_t randomCrossoverPoint = rand() % count;
		
		CPU_learn_crossOver(wOldFirst, wOldSecond,
							wNewFirst, wNewSecond,
							count, randomCrossoverPoint);
		CPU_learn_mutate(wNewFirst, count, m_mutationChance, m_mutationFactor);
		CPU_learn_mutate(wNewSecond, count, m_mutationChance, m_mutationFactor);
	}
	void GeneticNet::CPU_learn_crossOver(const float* wOldFirst,
										 const float* wOldSecond,
										 float* wNewFirst,
										 float* wNewSecond,
									     size_t count,
										 size_t crossoverPoint)
	{
		size_t cpySize = crossoverPoint * sizeof(float);
		memcpy(wNewFirst, wOldFirst, cpySize);
		memcpy(wNewSecond, wOldSecond, cpySize);

		size_t otherPartCount = count - crossoverPoint;
		wOldFirst += crossoverPoint;
		wOldSecond += crossoverPoint;
		wNewFirst += crossoverPoint;
		wNewSecond += crossoverPoint;

		cpySize = otherPartCount * sizeof(float);
		memcpy(wNewFirst, wOldFirst, cpySize);
		memcpy(wNewSecond, wOldSecond, cpySize);
	}
	void GeneticNet::CPU_learn_mutate(float* weights, size_t count,
								      float mutChance, float mutFac)
	{
		if (mutChance <= 0.000001)
			return;

		size_t iMutChance = 1.f / mutChance;

		for (size_t i = 0; i < count; ++i)
		{
			if (rand() % iMutChance == 0)
			{
				float randDeltaW = Net::getRandomValue(-mutFac, mutFac);
				weights[i] += randDeltaW;
			}
		}
	}



}
