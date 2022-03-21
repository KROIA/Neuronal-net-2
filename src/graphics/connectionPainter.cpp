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

			setConnectionWidth(m_standardConnectionWidth);
			setSignalWidth(m_standardSignalWidth);

			m_connectionLine.setPointCount(4);
			m_singalLine.setPointCount(4);
			
		}
		ConnectionPainter::~ConnectionPainter()
		{

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

		// Interface implementation
		void ConnectionPainter::draw(sf::RenderWindow* window,
							  const sf::Vector2f& offset)
		{
			

			m_weightColor.a = 100;
			m_signalColor.a = 100;

			m_connectionLine.setOrigin(-offset);
			m_singalLine.setOrigin(-offset);

			setLinePos(m_connectionLine, m_connectionWidth);
			setLinePos(m_singalLine, m_signalWidth);

			m_connectionLine.setFillColor(m_weightColor);
			m_singalLine.setFillColor(m_signalColor);

			m_signalColor = getColor(m_signal, m_globalMinSignal, m_globalMaxSignal);

			window->draw(m_connectionLine);
			window->draw(m_singalLine);

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

			m_weightColor = getColor(m_weight, minW, maxW);
			

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
