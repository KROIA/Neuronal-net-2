#pragma once


#include <iostream>
#include <SFML/Graphics.hpp>
#include <string>


#include "backend/utilities.h"
#include "backend/debug.h"
#include "backend/config.h"

namespace NeuronalNet
{
	namespace Graphics
	{
		enum VisualConfiguration
		{
			// Connections
			connectionWeights	= 1,
			connectionSignals	= 2,

			// Neurons
			neuronTextLabel		= 4,
			neuronBody   		= 8,

			// NetModell
			weightMap           = 16
		};
		NET_API extern sf::Color  getColor(float signal,
						   float min = -1.f,
						   float max =  1.f);
	};
};