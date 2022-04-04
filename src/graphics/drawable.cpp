#include "graphics/drawable.h"


namespace NeuronalNet
{
	namespace Graphics
	{

		Drawable::Drawable()
			: m_pos(0, 0)
			, m_visible(true)
			, m_drawDebug(false)
			//, m_optimization(Optimization::quality)
			, m_visualConfiguration(0)
		{
		}
		Drawable::Drawable(const Drawable& other)
			: m_pos(other.m_pos)
			, m_visible(other.m_visible)
			, m_drawDebug(other.m_drawDebug)
			//, m_optimization(Optimization::quality)
			, m_visualConfiguration(0)
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

		inline void Drawable::setVisible(bool enable)
		{
			m_visible = enable;
		}
		inline bool Drawable::isVisible() const
		{
			return m_visible;
		}
		inline void Drawable::setDrawDebug(bool enable)
		{
			m_drawDebug = enable;
		}
		inline bool Drawable::doesDrawDebug() const
		{
			return m_drawDebug;
		}

		/*void Drawable::setOptimization(Optimization opt)
		{
			m_optimization = opt;
		}
		Optimization Drawable::getOptimization() const
		{
			return m_optimization;
		}*/
		void Drawable::setVisualConfiguration(size_t conf)
		{
			m_visualConfiguration = conf;
		}
		inline size_t Drawable::getVisualConfiguration() const
		{
			return m_visualConfiguration;
		}
		/*inline size_t Drawable::getStandardVisualConfiguration() const
		{
			return 0;
		}*/

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