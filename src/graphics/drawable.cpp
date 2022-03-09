#include "graphics/drawable.h"


namespace NeuronalNet
{
	namespace Graphics
	{

		Drawable::Drawable()
			: m_pos(0, 0)
		{

		}
		Drawable::Drawable(const Drawable& other)
			: m_pos(other.m_pos)
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

		void Drawable::pos(const sf::Vector2f& pos)
		{
			m_pos = pos;
		}
		const sf::Vector2f& Drawable::pos() const
		{
			return m_pos;
		}

	};
};