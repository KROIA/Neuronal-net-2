#include "graphics/graphicsUtilities.h"

namespace NeuronalNet
{
	namespace Graphics
	{
		sf::Color getColor(float signal,
						   float min,
						   float max)
		{
			/*sf::Color color;
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
			return color;*/

			sf::Color color;
			color.b = 0;
			float x = signal *  3.14159265 / (max - min);
			color.g = (1.f + sin(x))*127.f;
			color.r = (1.f + sin(-x))*127.f;
			/*if (signal < 0)
			{
				if (signal < min)
					signal = min;
				color.g = 0.f;
				color.r = 255.f / min * signal;
			}
			else
			{
				if (signal > max)
					signal = max;
				color.r = 0.f;
				color.g = 255.f / max * signal;
			}*/
			return color;
		}
	};
};