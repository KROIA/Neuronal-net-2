#pragma once

#include <SFML/System.hpp>
#include "graphicsUtilities.h"

namespace NeuronalNet
{
	namespace Graphics
	{
		class NET_API DisplayInterface
		{
			public:
			virtual void onPreEventUpdate() = 0;
			virtual void onEvent(const sf::Event& event) = 0;
			virtual void onKeyEvent(const sf::Event& event) = 0;
			virtual void onKeyPressEvent(const sf::Event& event) = 0;
			virtual void onKeyReleaseEvent(const sf::Event& event) = 0;
			virtual void onMouseEvent(const sf::Event& event) = 0;
			virtual void onDisplyCloseEvent() = 0;
			virtual void onPostEventUpdate() = 0;

			virtual void onPreFrameUpdate() = 0;
			virtual void onPostFrameUpdate() = 0;

			virtual void onLoop() = 0;

		};
	};
};
