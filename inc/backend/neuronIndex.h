#pragma once

#include "config.h"
#include <string>

namespace NeuronalNet
{
	enum NeuronType
	{
		input,
		hidden,
		output,
		none
	};
	std::string NET_API typeToString(NeuronType t);

	struct NeuronIndex
	{
		NeuronType type;
		size_t x;
		size_t y;
	};

	struct ConnectionIndex
	{
		NeuronIndex neuron;
		size_t inputConnection;
	};
};