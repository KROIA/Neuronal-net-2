#include "graphics/connection.h"

namespace NeuronalNet
{
	namespace Graphics
	{
		const float Connection::standardConnectionWidth = 10;
		const float Connection::standardSignalWidth = 4;

		float Connection::m_globalMinWeight = -1.f;
		float Connection::m_globalMaxWeight =  1.f;
		Connection::Connection(Neuron* a,
							   Neuron* b)
			:GraphicsConnectionInterface()
		{
			m_startNeuron = a;
			m_endNeuron = b;

			connectionWidth(standardConnectionWidth);
			signalWidth(standardSignalWidth);

			m_connectionLine.setPointCount(4);
			m_singalLine.setPointCount(4);
			
		}
		Connection::~Connection()
		{

		}

		void Connection::connectionWidth(float w)
		{
			m_connectionWidth = w;
		}
		void Connection::signalWidth(float w)
		{
			m_signalWidth = w;
		}
		float Connection::connectionWidth() const
		{
			return m_connectionWidth;
		}
		float Connection::signalWidth() const
		{
			return m_signalWidth;
		}

		// Interface implementation
		void Connection::draw(sf::RenderWindow* window,
							  const sf::Vector2f& offset)
		{
			m_weightColor = getColor(m_weight, m_globalMinWeight, m_globalMaxWeight);
			m_signalColor = getColor(m_signal);

			m_weightColor.a = 100;
			m_signalColor.a = 100;

			m_connectionLine.setOrigin(-offset);
			m_singalLine.setOrigin(-offset);

			setLinePos(m_connectionLine, m_connectionWidth);
			setLinePos(m_singalLine, m_signalWidth);

			m_connectionLine.setFillColor(m_weightColor);
			m_singalLine.setFillColor(m_signalColor);



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

		void Connection::update(float weight, float signal)
		{
			m_weight = weight;
			m_signal = signal;

			// Update min/max Weight values
			if (m_weight < m_globalMinWeight)
				m_globalMinWeight = m_weight;
			else if (m_weight > m_globalMaxWeight)
				m_globalMaxWeight = m_weight;

			

			
		}

		void Connection::setLinePos(sf::ConvexShape& line, float width)
		{
			if (!(m_startNeuron || m_endNeuron))
				return;

			sf::Vector2f startPos = m_startNeuron->pos() + sf::Vector2f(m_startNeuron->size()-2,0);
			sf::Vector2f endPos   = m_endNeuron->pos() + sf::Vector2f(-m_endNeuron->size()+5,0);
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
