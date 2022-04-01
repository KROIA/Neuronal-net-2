#pragma once

#include "graphicsUtilities.h"

namespace NeuronalNet
{
	namespace Graphics
	{
		class DrawableInterface
		{
			public:
			virtual void displayCallDraw(sf::RenderWindow* window,
										 const sf::Vector2f& offset = sf::Vector2f(0, 0)) = 0;
		};
	};
};