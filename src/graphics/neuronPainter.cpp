#include "graphics/neuronPainter.h"
#include <string>

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
			m_outputText.setCharacterSize(m_size * 4 / 5);
			m_outputText.setFillColor(sf::Color(255, 255, 255));
			m_circleShape.setRadius(m_size);

			if(m_font.getInfo().family =="")
			if (!m_font.loadFromFile(m_fontPath))
			{
				CONSOLE("Can't load font: "<<m_fontPath)
			}
			m_outputText.setFont(m_font);

			//setOptimization(Optimization::quality);
			setVisualConfiguration(getStandardVisualConfiguration());

		}
		NeuronPainter::NeuronPainter(const NeuronPainter& other)
			: Drawable(other)
		{
			m_index			= other.m_index;
			m_color			= other.m_color;
			m_outputText	= other.m_outputText;
			m_font			= other.m_font;
			m_netinput		= other.m_netinput;
			m_output		= other.m_output;
			m_pos			= other.m_pos;
			m_size			= other.m_size;
			m_circleShape	= other.m_circleShape;
		}
		NeuronPainter::~NeuronPainter()
		{

		}

		const NeuronPainter& NeuronPainter::operator=(const NeuronPainter& other)
		{
			m_index				= other.m_index;
			m_color				= other.m_color;
			m_outputText		= other.m_outputText;
			m_font				= other.m_font;
			m_netinput			= other.m_netinput;
			m_output			= other.m_output;
			m_pos				= other.m_pos;
			m_size				= other.m_size;
			m_circleShape		= other.m_circleShape;
			return *this;
		}

		inline size_t NeuronPainter::getStandardVisualConfiguration()
		{
			return	VisualConfiguration::neuronTextLabel |
					VisualConfiguration::neuronBody;
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
			m_outputText.setCharacterSize(m_size * 4 / 5);
			m_circleShape.setRadius(m_size);
			m_circleShape.setOrigin(sf::Vector2f(m_size, m_size));
		}
		float NeuronPainter::getSize() const
		{
			return m_size;
		}

		/*void NeuronPainter::setOptimization(Optimization opt)
		{
			Drawable::setOptimization(opt);
			switch (m_optimization)
			{
				case Optimization::quality:
				{
					m_useTextLabel = true;
					break;
				}
				case Optimization::speed:
				{
					m_useTextLabel = false;
					break;
				}
				default:
				{
					PRINT_ERROR("Unknown optimization: "+std::to_string(opt))
				}
			}
		}*/

		void NeuronPainter::draw(sf::RenderWindow* window,
						  const sf::Vector2f &offset)
		{		
			if (m_visualConfiguration & VisualConfiguration::neuronBody)
			{
				m_circleShape.setPosition(m_pos + offset);
				window->draw(m_circleShape);
			}
			if (m_visualConfiguration & VisualConfiguration::neuronTextLabel)
			{
				m_outputText.setPosition(m_pos + offset);
				window->draw(m_outputText);
			}
		}

		// Interface implementation
		void NeuronPainter::update(float netinput, float output,
								   float minN, float maxN,
								   float minO, float maxO)
		{
			m_netinput	= netinput;
			m_output	= output;
			if (m_visualConfiguration & VisualConfiguration::neuronTextLabel)
			{
				char str[10];
				sprintf_s(str, "%6.3f", output);

                sf::Vector2f textSize((float)std::string(str).length() * m_outputText.getCharacterSize(),
									  m_outputText.getCharacterSize());

				m_outputText.setString(str);


				sf::FloatRect bound = m_outputText.getGlobalBounds();
				m_outputText.setOrigin(sf::Vector2f(bound.width / 2, bound.height / 2));
			}
			if (m_visualConfiguration & VisualConfiguration::neuronBody)
			{
				m_color = getColor(m_output, minO, maxO);
				m_circleShape.setFillColor(m_color);
			}
		}
	};
};
