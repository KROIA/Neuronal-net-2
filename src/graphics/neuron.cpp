#include "graphics/neuron.h"

namespace NeuronalNet
{
	namespace Graphics
	{
		const string Neuron::m_fontPath = "C:\\Windows\\Fonts\\arial.ttf";
			
		Neuron::Neuron()
			: Drawable()
		{
			m_color.b	= 0;
			m_color.a	= 255;
			
			m_index.type = NeuronType::none;
			m_index.x	= 0;
			m_index.y	= 0;

			m_netinput	= 0;
			m_output	= 0;

			m_pos = sf::Vector2f(0, 0);
			m_size = 20;

			if (!m_font.loadFromFile(m_fontPath))
			{
				CONSOLE("Can't load font: "<<m_fontPath)
			}
		}
		Neuron::Neuron(const Neuron& other)
			: Drawable(other)
		{
			m_index		= other.m_index;
			m_color		= other.m_color;
			m_outputText = other.m_outputText;
			m_font = other.m_font;
			m_netinput	= other.m_netinput;
			m_output	= other.m_output;
			m_pos = other.m_pos;
			m_size = other.m_size;
		}
		Neuron::~Neuron()
		{

		}

		const Neuron& Neuron::operator=(const Neuron& other)
		{
			m_index = other.m_index;
			m_color = other.m_color;
			m_outputText = other.m_outputText;
			m_font = other.m_font;
			m_netinput = other.m_netinput;
			m_output = other.m_output;
			m_pos = other.m_pos;
			m_size = other.m_size;
			return *this;
		}

		void Neuron::index(const NeuronIndex& index)
		{
			m_index = index;
		}
		void Neuron::pos(const sf::Vector2f& pos)
		{
			m_pos = pos;
		}
		const sf::Vector2f& Neuron::pos() const
		{
			return m_pos;
		}
		void Neuron::size(float size)
		{
			m_size = size;
		}
		float Neuron::size() const
		{
			return m_size;
		}

		void Neuron::draw(sf::RenderWindow* window,
						  const sf::Vector2f &offset)
		{
			sf::CircleShape shape(m_size);
			shape.setFillColor(m_color);
			shape.setPosition(m_pos + offset);

			m_outputText.setPosition(m_pos + offset);

			window->draw(shape);
			window->draw(m_outputText);
		}

		// Interface implementation
		void Neuron::update(float netinput, float output)
		{
			m_netinput	= netinput;
			m_output	= output;

			char str[10];
			sprintf_s(str, "%6.3f", output);
			m_outputText.setString(str);
			m_outputText.setCharacterSize(24);
			

			setColor(m_output);
		}
		const NeuronIndex& Neuron::index() const
		{
			return m_index;
		}


		void Neuron::setColor(float output)
		{
			if (output < 0)
			{
				if (output < -1.f)
					output = -1.f;
				m_color.g = 0;
				m_color.r = 255.f * output;
			}
			else
			{
				if (output > 1.f)
					output = 1.f;
				m_color.r = 0;
				m_color.g = 255.f * output;
			}
		}
	};
};