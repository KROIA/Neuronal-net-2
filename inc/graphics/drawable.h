#pragma once

#include "graphicsUtilities.h"
#include "drawableInterface.h"


namespace NeuronalNet
{
	namespace Graphics
	{

		class NET_API Drawable : public DrawableInterface
		{
			public:
			Drawable();
			Drawable(const Drawable& other);
			~Drawable();

			const Drawable& operator=(const Drawable& other);

			void setVisible(bool enable);
			bool isVisible() const;
			void setDrawDebug(bool enable);
			bool doesDrawDebug() const;

			void setPos(const sf::Vector2f& pos);
			void setPos(float x, float y);
			void move(const sf::Vector2f& deltaPos);
			void move(float dx, float dy);
			const sf::Vector2f& getPos() const;

			virtual void draw(sf::RenderWindow* window,
							  const sf::Vector2f& offset = sf::Vector2f(0, 0));
			virtual void drawDebug(sf::RenderWindow* window,
								   const sf::Vector2f& offset = sf::Vector2f(0, 0));

			void displayCallDraw(sf::RenderWindow* window,
								 const sf::Vector2f& offset = sf::Vector2f(0, 0));
			protected:

			sf::Vector2f m_pos;
			bool m_visible;
			bool m_drawDebug;
			private:

		};
	};
};