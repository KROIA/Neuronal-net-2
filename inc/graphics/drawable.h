#pragma once

#include <SFML/Graphics.hpp>

#include "backend/config.h"


namespace NeuronalNet
{
	namespace Graphics
	{

		class NET_API Drawable
		{
			public:
			Drawable();
			Drawable(const Drawable& other);
			~Drawable();

			const Drawable& operator=(const Drawable& other);

			void pos(const sf::Vector2f& pos);
			const sf::Vector2f &pos() const;

			virtual void draw(sf::RenderWindow* window) = 0;
			protected:

			private:
			sf::Vector2f m_pos;

		};
	};
};