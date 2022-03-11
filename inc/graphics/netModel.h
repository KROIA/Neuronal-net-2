#pragma once

#include <SFML/Graphics.hpp>
#include <vector>

#include "backend/net.h"
#include "neuron.h"
#include "connection.h"

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

			void rebuild();

			void streamIndex(size_t index);
			size_t streamIndex() const;

			void pos(const sf::Vector2f& pos);
			const sf::Vector2f& pos() const;

			void neuronSpacing(const sf::Vector2f& sp);
			const sf::Vector2f& neuronSpacing();

			void connectionWidth(float w);
			void signalWidth(float w);
			void neuronSize(float size);

			float connectionWidth() const;
			float signalWidth() const;
			float neuronSize() const;

			void draw(sf::RenderWindow* window,
					  const sf::Vector2f &offset = sf::Vector2f(0, 0));

			protected:
			void clear();
			void updateNeuronDimensions();
			//void internal_neuronSize(float size);

			vector<GraphicsNeuronInterface*> m_neuronInterface;
			vector<Neuron*> m_neuronList;
			vector<Neuron*> m_inputNeurons;
			vector<vector<Neuron*> > m_hiddenNeurons;
			vector<Neuron*> m_outputNeurons;

			vector<GraphicsConnectionInterface*> m_connectionInterface;
			vector<Connection*> m_connectionList;
			Net* m_net;

			float m_connectionWidth;
			float m_signalWidth;
			float m_neuronSize;
			size_t m_streamIndex;

			sf::Vector2f m_pos;
			sf::Vector2f m_neuronSpacing;
		};
	};
};