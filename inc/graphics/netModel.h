#pragma once

#include <SFML/Graphics.hpp>
#include <vector>

#include "backend/net.h"
#include "neuron.h"

namespace NeuronalNet
{
	namespace Graphics
	{
		using std::vector;

		class NET_API NetModel	: public Drawable
		{
			public:
			NetModel(Net* net);
			~NetModel();

			void rebuild();

			void draw(sf::RenderWindow* window,
					  const sf::Vector2f &offset = sf::Vector2f(0, 0));

			protected:
			void clear();

			vector<Neuron*> m_neuronList;
			Net* m_net;
		};
	};
};