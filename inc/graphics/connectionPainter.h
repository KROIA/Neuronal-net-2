#pragma once


#include "graphicsUtilities.h"
#include "backend/graphicsConnectionInterface.h"
#include "graphics/neuronPainter.h"


namespace NeuronalNet
{
	namespace Graphics
	{
		class ConnectionPainter : public Drawable, public GraphicsConnectionInterface
		{
			public:
			ConnectionPainter(NeuronPainter* a,
					   NeuronPainter* b);
			~ConnectionPainter();
			
			void setConnectionWidth(float w);
			void setSignalWidth(float w);
			float getConnectionWidth() const;
			float getSignalWidth() const;


			// Interface implementation
			void draw(sf::RenderWindow* window,
					  const sf::Vector2f& offset = sf::Vector2f(0, 0));

			void update(float weight, float signal,
						float minW, float maxW,
						float minS, float maxS);

			
			static float getStandardConnectionWidth();
			static float getStandardSignalWidth();

			protected:
			void setLinePos(sf::ConvexShape& line,float width);

			NeuronPainter* m_startNeuron;
			NeuronPainter* m_endNeuron;

			sf::Color m_weightColor;
			sf::Color m_signalColor;

			sf::ConvexShape m_connectionLine;
			sf::ConvexShape m_singalLine;

			float m_connectionWidth;
			float m_signalWidth;

			float m_weight;
			float m_signal;

			static float m_globalMinSignal;
			static float m_globalMaxSignal;

			static const float m_standardConnectionWidth;
			static const float m_standardSignalWidth;
		};
	};
};