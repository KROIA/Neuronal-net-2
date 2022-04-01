#pragma once


#include "graphicsUtilities.h"
#include "drawable.h"



namespace NeuronalNet
{
	namespace Graphics
	{
		class NET_API PixelPainter : public Drawable
		{
			public:
			PixelPainter();
			PixelPainter(const PixelPainter& other);
			~PixelPainter();

			void setDimenstions(const sf::Vector2u &dim);
			void setDimenstions(size_t x,size_t y);
			const sf::Vector2u& getDimensions();

			void setDisplaySize(const sf::Vector2f& size);
			void setDisplaySize(float x, float y);
			const sf::Vector2f& getDisplaySize() const;

			void setPixel(const sf::Vector2u& pos, const sf::Color& color);
			void setPixel(size_t x, size_t y, const sf::Color& color);
			const sf::Color& getPixel(const sf::Vector2u& pos);

			
			virtual void draw(sf::RenderWindow* window,
							  const sf::Vector2f& offset = sf::Vector2f(0, 0));
			virtual void drawDebug(sf::RenderWindow* window,
								   const sf::Vector2f& offset = sf::Vector2f(0, 0));


			protected:

			private:
			sf::Image *m_image;
			sf::Texture *m_texture;
			sf::Sprite *m_sprite;
			sf::Vector2f m_displaySize;
		};
	};
};