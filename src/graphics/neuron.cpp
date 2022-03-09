#include "graphics/neuron.h"

namespace NeuronalNet
{
	namespace Graphics
	{


			
		Neuron::Neuron()
			: Drawable()
		{

		}
		Neuron::Neuron(const Neuron& other)
			: Drawable(other)
		{

		}
		Neuron::~Neuron()
		{

		}

		const Neuron& Neuron::operator=(const Neuron& other)
		{

			return *this;
		}

		void Neuron::draw(sf::RenderWindow* window)
		{

		}
	};
};