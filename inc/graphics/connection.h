#pragma once


#include "graphicsUtilities.h"
#include "backend/graphicsConnectionInterface.h"
#include "graphics/neuron.h"


namespace NeuronalNet
{
	namespace Graphics
	{
		class Connection : public Drawable, public GraphicsConnectionInterface
		{
			public:
			Connection(Neuron* a,
					   Neuron* b);
			~Connection();
			
			void connectionWidth(float w);
			void signalWidth(float w);
			float connectionWidth() const;
			float signalWidth() const;


			// Interface implementation
			void draw(sf::RenderWindow* window,
					  const sf::Vector2f& offset = sf::Vector2f(0, 0));

			void update(float weight, float signal);

			static const float standardConnectionWidth;
			static const float standardSignalWidth;
			protected:
			void setLinePos(sf::ConvexShape& line,float width);

			Neuron* m_startNeuron;
			Neuron* m_endNeuron;

			sf::Color m_weightColor;
			sf::Color m_signalColor;

			sf::ConvexShape m_connectionLine;
			sf::ConvexShape m_singalLine;

			float m_connectionWidth;
			float m_signalWidth;

			float m_weight;
			float m_signal;

			static float m_globalMinWeight;
			static float m_globalMaxWeight;


		};
	};
};