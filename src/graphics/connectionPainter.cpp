#include "graphics/connectionPainter.h"

namespace NeuronalNet
{
	namespace Graphics
	{
		const float ConnectionPainter::m_standardConnectionWidth = 10;
		const float ConnectionPainter::m_standardSignalWidth = 4;

		float ConnectionPainter::m_globalMinSignal = -1.f;
		float ConnectionPainter::m_globalMaxSignal =  1.f;
		ConnectionPainter::ConnectionPainter(NeuronPainter* a,
							   NeuronPainter* b)
			:GraphicsConnectionInterface()
		{
			m_startNeuron = a;
			m_endNeuron = b;

			setWeightAlpha(100);
			setSignalAlpha(100);

			setConnectionWidth(m_standardConnectionWidth);
			setSignalWidth(m_standardSignalWidth);

			m_connectionLine.setPointCount(4);
			m_singalLine.setPointCount(4);

			//setOptimization(Optimization::quality);
			setVisualConfiguration(getStandardVisualConfiguration());
			
		}
		ConnectionPainter::~ConnectionPainter()
		{

		}

		inline size_t ConnectionPainter::getStandardVisualConfiguration()
		{
			return	VisualConfiguration::connectionSignals |
					VisualConfiguration::connectionWeights;
		}

		void ConnectionPainter::setConnectionWidth(float w)
		{
			m_connectionWidth = w;
		}
		void ConnectionPainter::setSignalWidth(float w)
		{
			m_signalWidth = w;
		}
		float ConnectionPainter::getConnectionWidth() const
		{
			return m_connectionWidth;
		}
		float ConnectionPainter::getSignalWidth() const
		{
			return m_signalWidth;
		}
		void ConnectionPainter::setWeightAlpha(uint8_t alpha)
		{
			m_weightColor.a = alpha;
		}
		uint8_t ConnectionPainter::getWeightAlpha() const
		{
			return m_weightColor.a;
		}
		void ConnectionPainter::setSignalAlpha(uint8_t alpha)
		{
			m_signalColor.a = alpha;
		}
		uint8_t ConnectionPainter::getSignalAlpha() const
		{
			return m_signalColor.a;
		}

		/*void ConnectionPainter::setOptimization(Optimization opt)
		{
			Drawable::setOptimization(opt);
			switch (m_optimization)
			{
				case Optimization::quality:
				{
					m_useWeightAsConnection = true;
					break;
				}
				case Optimization::speed:
				{
					m_useWeightAsConnection = false;
					break;
				}
				default:
				{
					PRINT_ERROR("Unknown optimization: " + std::to_string(opt))
				}
			}
		}*/

		// Interface implementation
		void ConnectionPainter::draw(sf::RenderWindow* window,
							  const sf::Vector2f& offset)
		{

			if (m_visualConfiguration & VisualConfiguration::connectionWeights)
			{
				m_connectionLine.setOrigin(-offset);
				setLinePos(m_connectionLine, m_connectionWidth);
				m_connectionLine.setFillColor(m_weightColor);
				window->draw(m_connectionLine);
			}

			if (m_visualConfiguration & VisualConfiguration::connectionSignals)
			{
				m_singalLine.setOrigin(-offset);
				setLinePos(m_singalLine, m_signalWidth);
				uint8_t alpha = m_signalColor.a;
				m_signalColor = getColor(m_signal, m_globalMinSignal, m_globalMaxSignal);
				m_signalColor.a = alpha;

				m_singalLine.setFillColor(m_signalColor);
				window->draw(m_singalLine);
			}

			/*sf::Vector2f startPos = m_startNeuron->pos() + sf::Vector2f(m_startNeuron->size() - 2, 0);
			sf::Vector2f endPos = m_endNeuron->pos() + sf::Vector2f(-m_endNeuron->size() + 5, 0);
			
			sf::Vertex line[] =
			{
				sf::Vertex(m_startNeuron->pos()),
				sf::Vertex(v1*10.f + m_startNeuron->pos()),

				sf::Vertex(startPos),
				sf::Vertex(endPos)
			};
			window->draw(line, 4, sf::Lines);*/
			
			
		}

		void ConnectionPainter::update(float weight, float signal,
									   float minW, float maxW,
									   float minS, float maxS)
		{

			m_weight = weight;
			m_signal = signal;

			uint8_t alpha = m_weightColor.a;
			if (m_visualConfiguration & VisualConfiguration::connectionWeights)
				m_weightColor = getColor(m_weight, minW, maxW);
			m_weightColor.a = alpha;

			// Update min/max Weight values
			if (m_weight < m_globalMinSignal)
				m_globalMinSignal = m_weight;
			else if (m_weight > m_globalMaxSignal)
				m_globalMaxSignal = m_weight;

			

			
		}

		float ConnectionPainter::getStandardConnectionWidth()
		{
			return m_standardConnectionWidth;
		}
		float ConnectionPainter::getStandardSignalWidth()
		{
			return m_standardSignalWidth;
		}

		void ConnectionPainter::setLinePos(sf::ConvexShape& line, float width)
		{
			if (!(m_startNeuron || m_endNeuron))
				return;

			sf::Vector2f startPos = m_startNeuron->getPos() + sf::Vector2f(m_startNeuron->getSize()-2,0);
			sf::Vector2f endPos   = m_endNeuron->getPos() + sf::Vector2f(-m_endNeuron->getSize()+5,0);
			sf::Vector2f deltaPos = endPos - startPos;


			deltaPos *= width / (2*sqrt(deltaPos.x * deltaPos.x + deltaPos.y * deltaPos.y)); // Norm and scale vector
			sf::Vector2f widthVec(deltaPos.y, -deltaPos.x); // Rotate vector 90 deg

			line.setPoint(0, startPos -  widthVec);
			line.setPoint(1, endPos   -  widthVec);
			line.setPoint(2, endPos   +  widthVec);
			line.setPoint(3, startPos +  widthVec);
		}




	};
};
