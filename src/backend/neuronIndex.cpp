#include "backend/neuronIndex.h"

namespace NeuronalNet
{
	std::string typeToString(NeuronType t)
	{
		switch (t) {
			case NeuronType::input:
				return "Input";
			case NeuronType::hidden:
				return "Hidden";
			case NeuronType::output:
				return "Output";
			default:
				return "none";
		}
	}
};