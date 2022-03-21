#include "graphics/pixelPainter.h"


namespace NeuronalNet
{
	namespace Graphics
	{
		PixelPainter::PixelPainter()
			: Drawable()
		{
			m_image = nullptr;
			m_texture = nullptr;
			m_sprite = nullptr;

			setDimenstions(sf::Vector2u(1, 1));
			setDisplaySize(sf::Vector2f(10,10));
		}

		PixelPainter::PixelPainter(const PixelPainter& other)
		{
			m_image = DBG_NEW sf::Image;
			m_texture = DBG_NEW sf::Texture(*other.m_texture);

			m_image->create(other.m_texture->getSize().x,
							other.m_texture->getSize().y);
			m_sprite = DBG_NEW sf::Sprite(*m_texture);
			setDisplaySize(other.m_displaySize);
		}

		PixelPainter::~PixelPainter()
		{
			if(m_sprite)
				delete m_sprite;
			if (m_texture)
				delete m_texture;
			if (m_image)
				delete m_image;
		}


		void PixelPainter::setDimenstions(const sf::Vector2u& dim)
		{
			setDimenstions(dim.x, dim.y);
		}
		void PixelPainter::setDimenstions(size_t x, size_t y)
		{
			if (m_sprite)
			{
				delete m_sprite;
				m_sprite = nullptr;
			}
			if (m_texture)
			{
				delete m_texture;
				m_texture = nullptr;
			}
			if (m_image)
			{
				delete m_image;
				m_image = nullptr;
			}

			m_image = DBG_NEW sf::Image;
			m_texture = DBG_NEW sf::Texture;

			m_image->create(x, y);
			m_texture->create(x, y);

			m_sprite = DBG_NEW sf::Sprite(*m_texture);
			m_sprite->setScale(m_displaySize);
		}

		const sf::Vector2u& PixelPainter::getDimensions()
		{
			return m_image->getSize();
		}

		void PixelPainter::setDisplaySize(const sf::Vector2f& size)
		{
			m_displaySize = size;
			m_sprite->setScale(m_displaySize);
		}
		void PixelPainter::setDisplaySize(float x, float y)
		{
			m_displaySize.x = x;
			m_displaySize.y = y;
			m_sprite->setScale(m_displaySize);
		}
		const sf::Vector2f& PixelPainter::getDisplaySize() const
		{
			return m_displaySize;
		}


		void PixelPainter::setPixel(const sf::Vector2u& pos, const sf::Color& color)
		{
			m_image->setPixel(pos.x, pos.y, color);
		}
		void PixelPainter::setPixel(size_t x, size_t y, const sf::Color& color)
		{
			m_image->setPixel(x, y, color);
		}

		const sf::Color& PixelPainter::getPixel(const sf::Vector2u& pos)
		{
			return m_image->getPixel(pos.x, pos.y);
		}



		void PixelPainter::draw(sf::RenderWindow* window,
								const sf::Vector2f& offset)
		{
			m_texture->loadFromImage(*m_image);
			m_sprite->setPosition(m_pos + offset);
			window->draw(*m_sprite);
		}

		void PixelPainter::drawDebug(sf::RenderWindow* window,
									 const sf::Vector2f& offset)
		{

		}

	}
}