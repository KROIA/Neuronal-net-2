#include "graphics/neuronPainter.h"

namespace NeuronalNet
{
	namespace Graphics
	{
		const float NeuronPainter::standardSize = 20;

		const string NeuronPainter::m_fontPath = "C:\\Windows\\Fonts\\arial.ttf";
		sf::Font NeuronPainter::m_font;
			
		NeuronPainter::NeuronPainter()
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
			m_size = standardSize;

			if(m_font.getInfo().family =="")
			if (!m_font.loadFromFile(m_fontPath))
			{
				CONSOLE("Can't load font: "<<m_fontPath)
			}
			m_outputText.setFont(m_font);
		}
		NeuronPainter::NeuronPainter(const NeuronPainter& other)
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
		NeuronPainter::~NeuronPainter()
		{

		}

		const NeuronPainter& NeuronPainter::operator=(const NeuronPainter& other)
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

		void NeuronPainter::setPos(const sf::Vector2f& pos)
		{
			m_pos = pos;
		}
		const sf::Vector2f& NeuronPainter::getPos() const
		{
			return m_pos;
		}
		void NeuronPainter::setSize(float size)
		{
			m_size = size;
		}
		float NeuronPainter::getSize() const
		{
			return m_size;
		}

		void NeuronPainter::draw(sf::RenderWindow* window,
						  const sf::Vector2f &offset)
		{
			sf::CircleShape shape(m_size);
			shape.setFillColor(m_color);
			shape.setOrigin(sf::Vector2f(m_size, m_size));
			shape.setPosition(m_pos + offset);
			

			m_outputText.setPosition(m_pos + offset);

			/*sf::FloatRect bound = m_outputText.getGlobalBounds();
			sf::Vertex line[] =
			{
				sf::Vertex(sf::Vector2f(bound.left,bound.top)),
				sf::Vertex(sf::Vector2f(bound.left+ bound.width,bound.top)),
				sf::Vertex(sf::Vector2f(bound.left + bound.width,bound.top+ bound.height)),
				sf::Vertex(sf::Vector2f(bound.left,bound.top + bound.height)),
				sf::Vertex(sf::Vector2f(bound.left,bound.top))
			};*/
			//window->draw(line, 5, sf::LineStrip);
			

			window->draw(shape);
			window->draw(m_outputText);

			
		}

		// Interface implementation
		void NeuronPainter::update(float netinput, float output,
								   float minN, float maxN,
								   float minO, float maxO)
		{
			m_netinput	= netinput;
			m_output	= output;

			char str[10];
			sprintf_s(str, "%6.3f", output);
			m_outputText.setCharacterSize(m_size*4 / 5);
			sf::Vector2f textSize((float)std::strlen(str) * m_outputText.getCharacterSize(),
								  m_outputText.getCharacterSize());
			//m_outputText.setOrigin(textSize / 2.f);
			m_outputText.setString(str);
			
			m_outputText.setFillColor(sf::Color(255, 255, 255));
			sf::FloatRect bound = m_outputText.getGlobalBounds();
			m_outputText.setOrigin(sf::Vector2f(bound.width / 2, bound.height / 2));

			m_color = getColor(m_output, minO, maxO);
		}
	};
};