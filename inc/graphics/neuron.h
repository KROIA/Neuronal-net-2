#pragma once

#include <SFML/Graphics.hpp>
#include "backend/config.h"
#include "drawable.h"

namespace NeuronalNet
{
	namespace Graphics
	{

		class NET_API Neuron	:	public Drawable
		{
			public:
			Neuron();
			Neuron(const Neuron& other);
			~Neuron();

			const Neuron& operator=(const Neuron& other);

			void draw(sf::RenderWindow* window);
			protected:

			private:

		};
	};
};







