#include "graphics/graphicsUtilities.h"

namespace NeuronalNet
{
	namespace Graphics
	{
		sf::Color getColor(float signal,
						   float min,
						   float max)
		{
			sf::Color color;
			if (signal < 0)
			{
				if (signal < min)
					signal = min;
				color.g = 0;
				color.r = 255.f / min * signal;
			}
			else
			{
				if (signal > max)
					signal = max;
				color.r = 0;
				color.g = 255.f / max * signal;
			}
			return color;
		}
	};
};