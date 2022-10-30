#pragma once

#include <SFML/Graphics.hpp>
#include <vector>

#include "backend/net.h"
#include "neuronPainter.h"
#include "connectionPainter.h"
#include "pixelPainter.h"

namespace NeuronalNet
{
	namespace Graphics
	{
		using std::vector;

		class NET_API NetModel	: public Drawable
		{
			public:

			

			NetModel(Net* net);
			~NetModel();

			void build();
			void setVisualConfiguration(size_t conf);
			static inline size_t getStandardVisualConfiguration();

			void setStreamIndex(size_t index);
			size_t getStreamIndex() const;

			void setPos(const sf::Vector2f& pos);
			const sf::Vector2f& getPos() const;

			void setNeuronSpacing(const sf::Vector2f& sp);
			const sf::Vector2f& getNeuronSpacing();

			void setConnectionWidth(float w);
			void setSignalWidth(float w);
			void setNeuronSize(float size);

			float getConnectionWidth() const;
			float getSignalWidth() const;
			float getNeuronSize() const;

			//void setOptimization(Optimization opt);
			//void setVisualConfiguration(size_t conf);

			void draw(sf::RenderWindow* window,
					  const sf::Vector2f &offset = sf::Vector2f(0, 0));
			void drawDebug(sf::RenderWindow* window,
						   const sf::Vector2f& offset = sf::Vector2f(0, 0));

			protected:
			void clear();
			void updateNeuronDimensions();
			void updateGraphics();
			//void internal_neuronSize(float size);

			vector<GraphicsNeuronInterface*> m_neuronInterface;	
			vector<NeuronPainter*> m_neuronList;					// All Neurons
			vector<NeuronPainter*> m_inputNeurons;					
			vector<vector<NeuronPainter*> > m_hiddenNeurons;
			vector<NeuronPainter*> m_outputNeurons;

			vector<GraphicsConnectionInterface*> m_connectionInterface;
			vector<ConnectionPainter*> m_connectionList;
			vector<PixelPainter*> m_pixelPainterList;

			vector<Drawable*> m_drawableList; // All drawable Objects

			const Net* m_net;

			float m_connectionWidth;
			float m_signalWidth;
			float m_neuronSize;
			size_t m_streamIndex;

			//sf::Vector2f m_pos;
			sf::Vector2f m_neuronSpacing;

			

		};
	};
};