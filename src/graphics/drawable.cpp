#include "graphics/drawable.h"


namespace NeuronalNet
{
	namespace Graphics
	{

		Drawable::Drawable()
			: m_pos(0, 0)
			, m_visible(true)
			, m_drawDebug(false)
		{
		}
		Drawable::Drawable(const Drawable& other)
			: m_pos(other.m_pos)
			, m_visible(other.m_visible)
			, m_drawDebug(other.m_drawDebug)
		{

		}
		Drawable::~Drawable()
		{

		}

		const Drawable& Drawable::operator=(const Drawable& other)
		{
			m_pos = other.m_pos;

			return *this;
		}

		void Drawable::setVisible(bool enable)
		{
			m_visible = enable;
		}
		bool Drawable::isVisible() const
		{
			return m_visible;
		}
		void Drawable::setDrawDebug(bool enable)
		{
			m_drawDebug = enable;
		}
		bool Drawable::doesDrawDebug() const
		{
			return m_drawDebug;
		}

		void Drawable::setPos(const sf::Vector2f& pos)
		{
			m_pos = pos;
		}
		void Drawable::setPos(float x, float y)
		{
			m_pos.x = x;
			m_pos.y = y;
		}
		void Drawable::move(const sf::Vector2f& deltaPos)
		{
			m_pos += deltaPos;
		}
		void Drawable::move(float dx, float dy)
		{
			m_pos.x += dx;
			m_pos.y += dy;
		}
		const sf::Vector2f& Drawable::getPos() const
		{
			return m_pos;
		}
		void Drawable::draw(sf::RenderWindow* window,
							const sf::Vector2f& offset)
		{

		}
		void Drawable::drawDebug(sf::RenderWindow* window,
								 const sf::Vector2f& offset)
		{

		}

		void Drawable::displayCallDraw(sf::RenderWindow* window,
									   const sf::Vector2f& offset)
		{
			if (m_visible)
			{
				draw(window, offset);
				if (m_drawDebug)
					drawDebug(window, offset);
			}
		}



	};
};