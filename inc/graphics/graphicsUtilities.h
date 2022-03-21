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
		NET_API extern sf::Color  getColor(float signal,
						   float min = -1.f,
						   float max =  1.f);
	};
};