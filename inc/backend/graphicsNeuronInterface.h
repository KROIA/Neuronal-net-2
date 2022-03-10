#pragma once
#include <string>
#include "config.h"

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
	class NET_API GraphicsNeuronInterface
	{
		public:

		virtual void update(float netinput, float output) = 0;

		const virtual NeuronIndex& index() const = 0;


		protected:
	};
};